# import required modules for project _NIDHI SHAH
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import plotly.express as px

# title of project
st.title("KPMG customer segmentation project by NIDHI SHAH")
# show the sidebar with radio buttons to select model for customer segmentation
selection = st.sidebar.radio("Select Customer segmentation base feature",["revenue aspect segmentation","product and revenue aspect segmentation","geolocation and revenue","purchase behaviour segmentation"])
# st.write(selection)

# read the csv file and create the main dataframe
cust_tran_df=pd.read_csv("C:/Users/nidhi/Documents/CPSC597/final_cust_tran_combine.csv")

# st.write(cust_tran_df)
# based on selection of feature, run the model - to run the model press the button
if st.sidebar.button("Run model"):
    
    if selection == "revenue aspect segmentation":
        st.write("Customer segmentation based on revenue, past_3_years_bike_related_purchases and Age")
        cust_rev_df=cust_tran_df.copy()
        st.text(f"Total customers are {cust_rev_df.shape [0]}")
        # select features to do customer segmentation
        seleted_features=cust_rev_df.loc[:,['past_3_years_bike_related_purchases','Age','Revenue']]
        
        # now apply the elbow method for selected features
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters = k, init = 'k-means++', max_iter = 300, n_init = 10)
            kmeans.fit(seleted_features)
            wcss.append(kmeans.inertia_)
        no_of_clusters = np.arange(1,11)
        fig1 = px.line(x=no_of_clusters, y=wcss, title="Elbow Method plot for revenue based clustering")
        fig1.update_layout(xaxis_title_text='Number of clusters',yaxis_title_text='Sum_of_squared_distances or Inertia',width=900,height=500)
        
        
        st.write("Cluster value selected as 4.")
        # now use k value and create K-means clusters
        final_kmeans = KMeans(n_clusters=4,init = 'k-means++')
        final_kmeans.fit(seleted_features)
        cust_cluster_result = final_kmeans.fit_predict(seleted_features)
        cust_rev_df['cluster_rev_4']=cust_cluster_result
        cust_rev_df["cluster_rev_4"] = cust_rev_df["cluster_rev_4"].astype(str)
        fig4= px.scatter_3d(cust_rev_df, x='Age', y='past_3_years_bike_related_purchases', z='Revenue',
                    color='cluster_rev_4',hover_name="name")
        st.plotly_chart(fig4)
        st.plotly_chart(fig1)
        cluster0=cust_rev_df.loc[cust_rev_df['cluster_rev_4'] == '0']
        st.text(f"Cluster 0 result with {cluster0.shape [0]} customers.")
        cluster0=cluster0[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster0)
        cluster1=cust_rev_df.loc[cust_rev_df['cluster_rev_4'] == '1']
        st.text(f"Cluster 1 result with {cluster1.shape [0]} customers.")
        cluster1=cluster1[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster1)
        cluster2=cust_rev_df.loc[cust_rev_df['cluster_rev_4'] == '2']
        st.text(f"Cluster 2 result with {cluster2.shape [0]} customers.")
        cluster2=cluster2[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster2)
        cluster3=cust_rev_df.loc[cust_rev_df['cluster_rev_4'] == '3']
        st.text(f"Cluster 3 result with {cluster3.shape [0]} customers.")
        cluster3=cluster3[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster3)
    ################################################################
    elif selection == "product and revenue aspect segmentation":
        st.write("Customer segmentation based on revenue and product purchased")
        cust_prod_df= cust_tran_df.copy()
        st.text(f"Total customers are {cust_prod_df.shape [0]}")
        # change the brand columns to numeric from categorical
        cust_prod_df['brand'].replace(['Solex','Trek Bicycles','OHM Cycles','Norco Bicycles','Giant Bicycles','WeareA2B'],
                        [0, 1,2,3,4,5], inplace=True)
        # st.write(cust_prod_df)
        # select features to do customer segmentation
        seleted_features_prod=cust_prod_df.loc[:,['brand','Revenue']]

        # now apply the elbow method for selected features
        wcss = []
        for k in range(1, 11):
            kmeans_prod= KMeans(n_clusters = k, init = 'k-means++', max_iter = 300, n_init = 10)
            kmeans_prod.fit(seleted_features_prod)
            wcss.append(kmeans_prod.inertia_)
        no_of_clusters = np.arange(1,11)
        fig5 = px.line(x=no_of_clusters, y=wcss, title="Elbow Method plot for product based clustering")
        fig5.update_layout(xaxis_title_text='Number of clusters',yaxis_title_text='Sum_of_squared_distances or Inertia',width=900,height=500)
       
        
        st.write("Cluster value selected as 3.")
        # now use k value and create K-means clusters
        final_kmeans_prod = KMeans(n_clusters=3,init = 'k-means++')
        final_kmeans_prod.fit(seleted_features_prod)
        cust_cluster_result_prod = final_kmeans_prod.fit_predict(seleted_features_prod)
        cust_prod_df['cluster_prod_3']=cust_cluster_result_prod
        cust_prod_df["cluster_prod_3"] = cust_prod_df["cluster_prod_3"].astype(str)
        cust_prod_df['brand'].replace([0, 1,2,3,4,5],
                        ['Solex','Trek Bicycles','OHM Cycles','Norco Bicycles','Giant Bicycles','WeareA2B'], inplace=True)
        # scatter plot for cluster data
        fig6 = px.scatter(cust_prod_df, x="brand", y="Revenue", color="cluster_prod_3",hover_name="name",hover_data=["brand","Revenue","past_3_years_bike_related_purchases"])
        fig6.update_traces(marker_size=10)
        fig6.update_coloraxes(showscale=False)
        st.plotly_chart(fig6)
        st.plotly_chart(fig5)
        # show clusters
        cluster0=cust_prod_df.loc[cust_prod_df['cluster_prod_3'] == '0']
        st.text(f"Cluster 0 result with {cluster0.shape [0]} customers.")
        cluster0=cluster0[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster0)
        cluster1=cust_prod_df.loc[cust_prod_df['cluster_prod_3'] == '1']
        st.text(f"Cluster 1 result with {cluster1.shape [0]} customers.")
        cluster1=cluster1[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster1)
        cluster2=cust_prod_df.loc[cust_prod_df['cluster_prod_3'] == '2']
        st.text(f"Cluster 2 result with {cluster2.shape [0]} customers.")
        cluster2=cluster2[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster2)

    #########################################################################################
    elif selection == "geolocation and revenue":
        st.write("Customer segmentation based on revenue and location of customers- states of Australia")
        cust_state_df= cust_tran_df.copy()
        st.text(f"Total customers are {cust_state_df.shape [0]}")
        # change the state columns to numeric from categorical
        cust_state_df['state'].replace(['NSW','VIC','QLD'],
                            [0,1,2], inplace=True)
        
        # select features to do customer segmentation
        seleted_features_state=cust_state_df.loc[:,['state','Revenue']]
        # now apply the elbow method for selected features
        wcss = []
        for k in range(1, 11):
            kmeans_state= KMeans(n_clusters = k, init = 'k-means++', max_iter = 300, n_init = 10)
            kmeans_state.fit(seleted_features_state)
            wcss.append(kmeans_state.inertia_)
        no_of_clusters = np.arange(1,11)
        fig7 = px.line(x=no_of_clusters, y=wcss, title="Elbow Method plot for state based clustering")
        fig7.update_layout(xaxis_title_text='Number of clusters',yaxis_title_text='Sum_of_squared_distances or Inertia',width=900,height=500)
       

        st.write("Cluster value selected as 3.")
        # now use k value and create K-means clusters
        final_kmeans_state = KMeans(n_clusters=3,init = 'k-means++')
        final_kmeans_state.fit(seleted_features_state)
        cust_cluster_result_state = final_kmeans_state.fit_predict(seleted_features_state)
        cust_state_df['cluster_state_3']=cust_cluster_result_state
        cust_state_df["cluster_state_3"] = cust_state_df["cluster_state_3"].astype(str)
        cust_state_df['state'].replace([0,1,2],
                        ['NSW','VIC','QLD'], inplace=True)
        # scatter plot for cluster data
        fig8 = px.scatter(cust_state_df, x="state", y="Revenue", color="cluster_state_3",hover_name="name",hover_data=["full_address","brand","product_class","product_line","product_size"])
        fig8.update_traces(marker_size=10)
        fig8.update_coloraxes(showscale=False)
        st.plotly_chart(fig8)
        st.plotly_chart(fig7)
        # show clusters
        cluster0=cust_state_df.loc[cust_state_df['cluster_state_3'] == '0']
        st.text(f"Cluster 0 result with {cluster0.shape [0]} customers.")
        cluster0=cluster0[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster0)
        cluster1=cust_state_df.loc[cust_state_df['cluster_state_3'] == '1']
        st.text(f"Cluster 1 result with {cluster1.shape [0]} customers.")
        cluster1=cluster1[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster1)
        cluster2=cust_state_df.loc[cust_state_df['cluster_state_3'] == '2']
        st.text(f"Cluster 2 result with {cluster2.shape [0]} customers.")
        cluster2=cluster2[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster2)

    #######################################################################
    elif selection=="purchase behaviour segmentation":
        st.write("Customer segmentation based on past_3_years_bike_related_purchases and revenue")
        cust_pur_df= cust_tran_df.copy()
        st.text(f"Total customers are {cust_pur_df.shape [0]}")
        
        # select features to do customer segmentation
        seleted_features_pur=cust_pur_df.loc[:,['past_3_years_bike_related_purchases','Revenue']]
        # now apply the elbow method for selected features
        wcss = []
        for k in range(1, 11):
            kmeans_pur= KMeans(n_clusters = k, init = 'k-means++', max_iter = 300, n_init = 10)
            kmeans_pur.fit(seleted_features_pur)
            wcss.append(kmeans_pur.inertia_)
        no_of_clusters = np.arange(1,11)
        fig9 = px.line(x=no_of_clusters, y=wcss, title="Elbow Method plot for purchase behavior based clustering")
        fig9.update_layout(xaxis_title_text='Number of clusters',yaxis_title_text='Sum_of_squared_distances or Inertia',width=900,height=500)
        

        st.write("Cluster value selected as 3.")
        # now use k value and create K-means clusters
        final_kmeans_pur = KMeans(n_clusters=3,init = 'k-means++')
        final_kmeans_pur.fit(seleted_features_pur)
        cust_cluster_result_pur = final_kmeans_pur.fit_predict(seleted_features_pur)
        cust_pur_df['cluster_state_3']=cust_cluster_result_pur
        cust_pur_df["cluster_state_3"] = cust_pur_df["cluster_state_3"].astype(str)
        cust_pur_df['state'].replace([0,1,2],
                        ['NSW','VIC','QLD'], inplace=True)
        # scatter plot for cluster data
        fig10 = px.scatter(cust_pur_df, y="past_3_years_bike_related_purchases", x="Revenue", color="cluster_state_3",hover_name="name",hover_data=["full_address","brand","product_class","product_line","product_size"])
        fig10.update_traces(marker_size=10)
        fig10.update_coloraxes(showscale=False)
        st.plotly_chart(fig10)
        st.plotly_chart(fig9)
        # show clusters
        cluster0=cust_pur_df.loc[cust_pur_df['cluster_state_3'] == '0']
        st.text(f"Cluster 0 result with {cluster0.shape [0]} customers.")
        cluster0=cluster0[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster0)
        cluster1=cust_pur_df.loc[cust_pur_df['cluster_state_3'] == '1']
        st.text(f"Cluster 1 result with {cluster1.shape [0]} customers.")
        cluster1=cluster1[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster1)
        cluster2=cust_pur_df.loc[cust_pur_df['cluster_state_3'] == '2']
        st.text(f"Cluster 2 result with {cluster2.shape [0]} customers.")
        cluster2=cluster2[['name','gender','Age','past_3_years_bike_related_purchases','transaction_date','Revenue','brand','product_class','full_address']]
        st.dataframe(cluster2)