#cloning github repository
!git clone "https://github.com/cereniyim/Customer-Segmentation-Unsupervised-ML-Model"

# data wrangling
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# for data preprocessing and clustering
from sklearn.cluster import KMeans

%matplotlib inline
# to include graphs inline within the frontends next to code

%config InlineBackend.figure_format='retina'
#to enable retina (high resolution) plots

pd.options.mode.chained_assignment = None
# to bypass warnings in various dataframe assignments

#read the data
data = pd.read_csv("/content/Customer-Segmentation-Unsupervised-ML-Model/Orders - Analysis Task.csv")
print(data.head())

print(data[data.net_quantity < 0].shape[0])

data = data[data["ordered_item_quantity"]>0]

def encode_column(column):
  if column > 0:
    return 1
  if column <= 0:
    return 0
def aggregate_by_ordered_quantity(dataframe, column_list):
    aggregated_dataframe = (dataframe
                            .groupby(column_list)
                            .ordered_item_quantity.count()
                            .reset_index())
    aggregated_dataframe["products_ordered"] = (aggregated_dataframe
                                              .ordered_item_quantity
                                              .apply(encode_column))
    final_dataframe = (aggregated_dataframe
                       .groupby(column_list[0])
                       .products_ordered.sum() # aligned with the added column name
                       .reset_index())

    return final_dataframe

customers = aggregate_by_ordered_quantity(data, ["customer_id", "product_type"])
print(customers.head())


# aggregate data per customer_id and order_id, 
# to see ordered item sum and returned item sum
ordered_sum_by_customer_order = (data
                                 .groupby(["customer_id", "order_id"])
                                 .ordered_item_quantity.sum()
                                 .reset_index())

returned_sum_by_customer_order = (data
                                  .groupby(["customer_id", "order_id"])
                                  .returned_item_quantity.sum()
                                  .reset_index())

# merge two dataframes to be able to calculate unit return rate
ordered_returned_sums = pd.merge(ordered_sum_by_customer_order, returned_sum_by_customer_order)

ordered_returned_sums["average_return_rate"] = (-1 * 
                                             ordered_returned_sums["returned_item_quantity"] /
                                             ordered_returned_sums["ordered_item_quantity"])
ordered_returned_sums.head()


# take average of the unit return rate for all orders of a customer
customer_return_rate = (ordered_returned_sums
                        .groupby("customer_id")
                        .average_return_rate
                        .mean()
                        .reset_index())
return_rates = pd.DataFrame(customer_return_rate["average_return_rate"]
                            .value_counts()
                            .reset_index())

return_rates.rename(columns=
                    {"index": "average return rate",
                     "average_return_rate": "count of unit return rate"},
                    inplace=True)

return_rates.sort_values(by="average return rate")


# add average_return_rate to customers dataframe
customers = pd.merge(customers,
                     customer_return_rate,
                     on="customer_id")

# aggreagate total sales per customer id
customer_total_spending = (data
                           .groupby("customer_id")
                           .total_sales
                           .sum()
                           .reset_index())

customer_total_spending.rename(columns = {"total_sales" : "total_spending"},
                               inplace = True)

# add total sales to customers dataframe
customers = customers.merge(customer_total_spending, 
                            on="customer_id")
print("The number of customers from the existing customer base:", customers.shape[0])


# drop id column since it is not a feature
customers.drop(columns="customer_id",
               inplace=True)
customers.head()


fig = make_subplots(rows=3, cols=1,
                   subplot_titles=("Products Ordered", 
                                   "Average Return Rate", 
                                   "Total Spending"))

fig.append_trace(go.Histogram(x=customers.products_ordered),
                 row=1, col=1)

fig.append_trace(go.Histogram(x=customers.average_return_rate),
                 row=2, col=1)

fig.append_trace(go.Histogram(x=customers.total_spending),
                 row=3, col=1)

fig.update_layout(height=800, width=800,
                  title_text="Distribution of the Features")

fig.show()




def apply_log1p_transformation(dataframe, column):
    '''This function takes a dataframe and a column in the string format
    then applies numpy log1p transformation to the column
    as a result returns log1p applied pandas series'''
    
    dataframe["log_" + column] = np.log1p(dataframe[column])
    return dataframe["log_" + column]

apply_log1p_transformation(customers, "products_ordered")

apply_log1p_transformation(customers, "average_return_rate")

apply_log1p_transformation(customers, "total_spending")




fig = make_subplots(rows=3, cols=1,
                   subplot_titles=("Products Ordered", 
                                   "Average Return Rate", 
                                   "Total Spending"))

fig.append_trace(go.Histogram(x=customers.log_products_ordered),
                 row=1, col=1)

fig.append_trace(go.Histogram(x=customers.log_average_return_rate),
                 row=2, col=1)

fig.append_trace(go.Histogram(x=customers.log_total_spending),
                 row=3, col=1)

fig.update_layout(height=800, width=800,
                  title_text="Distribution of the Features after Logarithm Transformation")

fig.show()





customers.head()

# features we are going to use as an input to the model
customers.iloc[:,3:]



# create initial K-means model
kmeans_model = KMeans(init='k-means++', 
                      max_iter=500, 
                      random_state=42)
kmeans_model.fit(customers.iloc[:,3:])

# print the sum of distances from all examples to the center of the cluster
print("within-cluster sum-of-squares (inertia) of the model is:", kmeans_model.inertia_)



def make_list_of_K(K, dataframe):
    '''inputs: K as integer and dataframe
    apply k-means clustering to dataframe
    and make a list of inertia values against 1 to K (inclusive)
    return the inertia values list
    '''
    
    cluster_values = list(range(1, K+1))
    inertia_values=[]
    
    for c in cluster_values:
        model = KMeans(
            n_clusters = c, 
            init='k-means++', 
            max_iter=500, 
            random_state=42)
        model.fit(dataframe)
        inertia_values.append(model.inertia_)
    
    return inertia_values


# save inertia values in a dataframe for k values between 1 to 15
results = make_list_of_K(15, customers.iloc[:, 3:])

k_values_distances = pd.DataFrame({"clusters": list(range(1, 16)),
                                   "within cluster sum of squared distances": results})



# visualization for the selection of number of segments
fig = go.Figure()

fig.add_trace(go.Scatter(x=k_values_distances["clusters"],
                         y=k_values_distances["within cluster sum of squared distances"],
                         mode='lines+markers'))
fig.update_layout(xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1),
                  title_text="Within Cluster Sum of Squared Distances VS K Values",
                  xaxis_title="K values",
                  yaxis_title="Cluster sum of squared distances")

fig.show()


# create clustering model with optimal k=4
updated_kmeans_model = KMeans(n_clusters = 4, 
                              init='k-means++', 
                              max_iter=500, 
                              random_state=42)

updated_kmeans_model.fit_predict(customers.iloc[:,3:])



# create cluster centers and actual data arrays
cluster_centers = updated_kmeans_model.cluster_centers_
actual_data = np.expm1(cluster_centers)
add_points = np.append(actual_data, cluster_centers, axis=1)
add_points

# add labels to customers dataframe and add_points array
add_points = np.append(add_points, [[0], [1], [2], [3]], axis=1)
customers["clusters"] = updated_kmeans_model.labels_



# create centers dataframe from add_points
centers_df = pd.DataFrame(data=add_points, columns=["products_ordered",
                                                    "average_return_rate",
                                                    "total_spending",
                                                    "log_products_ordered",
                                                    "log_average_return_rate",
                                                    "log_total_spending",
                                                    "clusters"])
centers_df.head()



# align cluster centers of centers_df and customers
centers_df["clusters"] = centers_df["clusters"].astype("int")
centers_df.head()



customers.head()



# differentiate between data points and cluster centers
customers["is_center"] = 0
centers_df["is_center"] = 1

# add dataframes together
customers = customers.append(centers_df, ignore_index=True)


customers.tail()



# add clusters to the dataframe
customers["cluster_name"] = customers["clusters"].astype(str)




# visualize log_transformation customer segments with a 3D plot
fig = px.scatter_3d(customers,
                    x="log_products_ordered",
                    y="log_average_return_rate",
                    z="log_total_spending",
                    color='cluster_name',
                    hover_data=["products_ordered",
                                "average_return_rate",
                                "total_spending"],
                    category_orders = {"cluster_name": 
                                       ["0", "1", "2", "3"]},
                    symbol = "is_center"
                    )

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()



# values for log_transformation
cardinality_df = pd.DataFrame(
    customers.cluster_name.value_counts().reset_index())

cardinality_df.rename(columns={"index": "Customer Groups",
                               "cluster_name": "Customer Group Magnitude"},
                      inplace=True)




cardinality_df




fig = px.bar(cardinality_df, x="Customer Groups", 
             y="Customer Group Magnitude",
             color = "Customer Groups",
             category_orders = {"Customer Groups": ["0","1","2","3"]})

fig.update_layout(xaxis = dict(
        tickmode = 'linear',
        tick0 = 1,
        dtick = 1),
        yaxis = dict(
        tickmode = 'linear',
        tick0 = 1000,
        dtick = 1000))
fig.show()

