import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from streamlit_folium import folium_static
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from twilio.rest import Client
import altair as alt
from bs4 import BeautifulSoup
import requests

# Custom CSS for full-screen background, fade-in effect, and other styling
def set_bg_as_image():
    st.markdown(
        """
        <style>
        /* Fullscreen pseudo-element with the background image */
        .stApp::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
            background-image: url("https://cdna.artstation.com/p/assets/images/images/016/265/566/original/mikhail-gorbunov-ui-1.gif?1551525784");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            filter: blur(8px);
        }

        /* Override Streamlit's default styling */
        .stApp {
            background-color: transparent;
        }

        /* Fade-in animation */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Fade-out animation */
        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }

        /* Styling for text and content box */
        h1, .content {
            color: #ffffff;
            z-index: 1;
            position: relative;
            animation: fadeIn 2s ease-in-out;
        }

        .content {
            padding: 2rem;
            background-color: rgba(0, 0, 0, 0.5);
            border-radius: 10px;
            margin-top: 4rem;
        }

        /* Styling for the sidebar */
        .css-1lcbmhc {
            top: 0;
            position: fixed;
            z-index: 2;
        }
        
        /* Responsive styles */
        @media (max-width: 640px) {
            /* Fullscreen pseudo-element with the background image */
            .stApp::before {
                background-size: contain;
                background-position: top;
            }

            /* Content adjustments */
            .content {
                padding: 1rem; /* smaller padding */
                margin-top: 2rem; /* smaller margin */
                font-size: 14px; /* smaller font size */
            }

            /* Sidebar adjustments */
            .css-1lcbmhc, .stSidebar {
                width: 100% !important;
                z-index: 2;
            }

            /* Hide the hamburger menu to save space */
            .css-1v3fvcr {
                display: none;
            }

            /* Adjust specific Streamlit elements for smaller screens */
            .stButton > button, .stTextInput > div > div > input, .stSelectbox > select {
                width: 100% !important;
                font-size: 16px; /* Adjust font size as needed */
            }

            /* Make plotly charts responsive */
            .plotly-graph-div {
                width: 100% !important;
            }

            /* Adjust the title for smaller screens */
            h1 {
                font-size: 22px; /* Smaller font size for the title */
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background
set_bg_as_image()

# Remove the file uploader and read CSV directly from URL
csv_url = 'https://raw.githubusercontent.com/DevWithJas/Sentinel-Mark-2/main/crime-sentinel.csv.csv'

# Caching the data loading to optimize memory usage
@st.cache_data
def load_data():
    df = pd.read_csv(csv_url)
    return df

df = load_data()

# Display the heading
st.title("Welcome to Sentinel Mark 2 : Crime Predictive Model ")

# Emoji for large headings
crime_chart_emoji = "📊"
intensity_map_emoji = "🗺️"

# Correct column names based on your CSV file
crime_types = ['murder', 'rape', 'gangrape', 'robbery', 'theft', 'assualt murders', 'sexual harassement']
df_crime_types = df[crime_types + ['nm_pol', 'long', 'lat']]
df_grouped = df_crime_types.groupby('nm_pol').sum().reset_index()
df_grouped['Total Crime Intensity'] = df_grouped[crime_types].sum(axis=1)

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    'Select a category:',
    (
        'Home',
        'Intensity and Chart',
        'Crime Map',
        'Custom Location Prediction',
        'Search',
        'Crime Against Women',
        'Crime News',
        'Data Relation Chart',
        'Report Suspicious Activity',
        'Safety Tips'
    )
)

# Home Page Content
if option == 'Home':
    st.markdown('''
        <div class="content">
            <p>"Sentinel Mark 2, the successor of its previous iteration, stands as a robust crime analysis and prediction tool. Engineered with precision, it harnesses cutting-edge analytics to forecast criminal activity with startling accuracy. This advanced tool aids law enforcement agencies in preemptive measures, ensuring public safety with proactive strategies. Its seamless integration with modern tech provides an unparalleled edge in the realm of crime prevention."</p>
        </div>
    ''', unsafe_allow_html=True)

# Intensity and Chart Option
elif option == 'Intensity and Chart':
    # Display Crime Type Chart
    st.subheader(f'{crime_chart_emoji} Crime Type Chart')
    df_melted = pd.melt(df_grouped, id_vars=['nm_pol'], value_vars=crime_types, var_name='Crime Type', value_name='Count')
    fig = px.bar(
        df_melted,
        x='nm_pol',
        y='Count',
        color='Crime Type',
        title=f'{crime_chart_emoji} Crime Distribution by Police Station',
        labels={'nm_pol': 'Police Station', 'Count': 'Number of Incidents'}
    )
    fig.update_layout(
        xaxis_title="Police Station",
        yaxis_title="Number of Incidents",
        barmode='stack',
        xaxis={'categoryorder': 'total descending'},
        legend_title_text='Crime Type'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display the Crime Intensity Map
    st.subheader(f'{intensity_map_emoji} Crime Intensity Map')
    fig_map = px.scatter_mapbox(
        df_grouped,
        lat='lat',
        lon='long',
        size='Total Crime Intensity',
        color='Total Crime Intensity',
        hover_name='nm_pol',
        title=f'{intensity_map_emoji} Crime Intensity Map',
        mapbox_style="open-street-map",
        zoom=10
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

#<------------------------------------------------- Random Forest and Crime Map ------------------------------------------------------------------>
elif option == 'Crime Map':
    st.subheader(f'{intensity_map_emoji} Crime Probability Map')
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(df[['lat', 'long']], df[crime_types], test_size=0.3, random_state=42)
    class_weights = 'balanced'

    # Caching the model training to optimize memory usage
    @st.cache_resource
    def train_random_forest(X_train, y_train):
        multi_output_rf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
        multi_output_rf.fit(X_train, y_train)
        return multi_output_rf

    multi_output_rf = train_random_forest(X_train_rf, y_train_rf)

    y_pred_proba = multi_output_rf.predict_proba(X_test_rf)
    y_pred_proba_positive = np.array([proba[:, 1] for proba in y_pred_proba]).T

    map_center = [X_test_rf['lat'].mean(), X_test_rf['long'].mean()]
    crime_map_rf = folium.Map(location=map_center, zoom_start=12)
    high_probability_threshold = 0.5
    moderate_probability_threshold = 0.2

    for idx, (lat, long) in enumerate(zip(X_test_rf['lat'], X_test_rf['long'])):
        max_probability = y_pred_proba_positive[idx, :].max()
        if max_probability >= high_probability_threshold:
            color = 'red'
        elif max_probability >= moderate_probability_threshold:
            color = 'yellow'
        else:
            color = 'green'
        popup_info = '<br>'.join([f"{crime}: {y_pred_proba_positive[idx, i]:.2f}" for i, crime in enumerate(crime_types)])
        folium.CircleMarker(
            location=[lat, long],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            popup=folium.Popup(popup_info, max_width=300)
        ).add_to(crime_map_rf)
    folium_static(crime_map_rf)

# Custom Location Prediction Option
elif option == 'Custom Location Prediction':
    st.header("🔍 Get Crime Prediction for a Specific Location")

    # User Input for Latitude and Longitude
    user_lat = st.number_input("Enter Latitude:", value=df['lat'].mean())
    user_long = st.number_input("Enter Longitude:", value=df['long'].mean())

    # Prepare data for prediction based on CSV data
    if st.button("Predict Crime"):
        user_input = pd.DataFrame({'lat': [user_lat], 'long': [user_long]})
        y_pred_proba_user = multi_output_rf.predict_proba(user_input)
        y_pred_proba_positive_user = np.array([proba[:, 1] for proba in y_pred_proba_user]).T
        predicted_crimes = dict(zip(crime_types, y_pred_proba_positive_user[0]))
        st.write(f"Predicted Crime Probabilities at ({user_lat}, {user_long}):")
        for crime, prob in predicted_crimes.items():
            st.write(f"- **{crime.title()}**: {prob:.2f} probability")

# Search Functionality Option
elif option == 'Search':
    st.header("🔎 Search for a Location")
    search_query = st.text_input("Enter Police Station or Area Name:")
    if search_query:
        filtered_df = df[df['nm_pol'].str.contains(search_query, case=False)]
        if not filtered_df.empty:
            st.write(f"### Search Results for '{search_query}':")
            st.dataframe(filtered_df)
            # Update charts or maps with filtered data
            df_grouped_filtered = filtered_df.groupby('nm_pol').sum().reset_index()
            df_grouped_filtered['Total Crime Intensity'] = df_grouped_filtered[crime_types].sum(axis=1)
            df_melted_filtered = pd.melt(df_grouped_filtered, id_vars=['nm_pol'], value_vars=crime_types, var_name='Crime Type', value_name='Count')
            fig_filtered = px.bar(
                df_melted_filtered,
                x='nm_pol',
                y='Count',
                color='Crime Type',
                title=f'{crime_chart_emoji} Crime Distribution for Search Results',
                labels={'nm_pol': 'Police Station', 'Count': 'Number of Incidents'}
            )
            st.plotly_chart(fig_filtered, use_container_width=True)
        else:
            st.warning("No results found. Please try a different search term.")
    else:
        st.write("Please enter a search term to find specific locations.")

#<-------------------------------------------- Crime Against Women Delhi Police Data ------------------------------------------------------------------>
elif option == 'Crime Against Women':
    # Data manually extracted from the image
    data_caw = {
        'Year': ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'],
        'RAPE': [706, 1636, 2166, 2199, 2153, 2168, 1699, 2076, 1033, 1100],
        'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY': [727, 3515, 4322, 5367, 3314, 2921, 2186, 2551, 1244, 1480],
        'INSULT TO THE MODESTY OF WOMEN': [214, 916, 1361, 1941, 599, 495, 434, 440, 229, 225],
        'KIDNAPPING OF WOMEN': [2048, 3286, 3604, 3738, 3482, 3471, 3761, 3758, 1880, 2197],
        'ABDUCTION OF WOMEN': [162, 323, 423, 556, 262, 201, 177, 325, 184, 105],
        'CRUELTY BY HUSBAND AND IN LAWS': [2046, 3045, 3194, 3536, 3416, 3792, 4557, 4731, 2096, 2704],
        'DOWRY DEATH': [134, 144, 153, 153, 116, 110, 141, 72, 69, 69],
        'DOWRY PROHIBITION ACT': [15, 15, 13, 20, 26, 16, 14, 16, 9, 7]
    }

    # Convert the dictionary to a pandas DataFrame and melt it for Plotly
    df_caw = pd.DataFrame(data_caw)
    df_caw = df_caw.melt(id_vars=['Year'], var_name='Crime', value_name='Cases')

    # Create the Plotly figure
    fig_caw = px.bar(df_caw, x='Year', y='Cases', color='Crime', title='Crime Against Women from 2012 to 2021')
    fig_caw.update_layout(barmode='group')

    # Display the Plotly chart
    st.plotly_chart(fig_caw)

#<-------------------------------------- News Scraping ------------------------------------------------------------------>
elif option == 'Crime News':
    st.header("Latest Crime News")

    # Add the function to scrape news
    def fetch_crime_news():
        url = 'https://www.ndtv.com/topic/delhi-crime'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        news_containers = soup.find_all('div', class_='src_itm-ttl')
        news_articles = []
        for container in news_containers:
            news_link = container.find('a')
            if news_link:
                title = news_link.text.strip()
                link = news_link['href'].strip()
                news_articles.append({'title': title, 'url': link})
        return news_articles

    news_data = fetch_crime_news()
    for article in news_data:
        st.markdown(f"### [{article['title']}]({article['url']})")
        st.markdown("---")

#<----------------------------------------- Data Relation Chart ------------------------------------------------------------------>
elif option == 'Data Relation Chart':
    st.header("Data Relation Chart")
    data = df.copy()

    # Remove "lat" and "long" columns if they exist
    df_chart = data.drop(["lat", "long"], axis=1, errors="ignore")

    # Select x and y-axis columns in the sidebar
    x_column = st.sidebar.selectbox("Select X-axis Column", df_chart.columns, key="x_column")
    y_column = st.sidebar.selectbox("Select Y-axis Column", df_chart.columns, key="y_column")

    # Check if exactly two columns are selected
    if x_column != y_column:
        # Scatter plot to compare the selected columns with custom colors
        st.write("### Data Relation Chart:")
        chart = alt.Chart(df_chart).mark_circle().encode(
            x=x_column,
            y=y_column,
            color=alt.Color("count()", scale=alt.Scale(range=['#FF69B4', '#ADD8E6'])),
            tooltip=[x_column, y_column, "count()"]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Please select two different columns for comparison.")

#<---------------------------------------------------- Report Suspicious Activity --------------------------------------->
elif option == 'Report Suspicious Activity':
    st.header("Report Suspicious Activity Anonymously")
    report_text = st.text_area("Describe the suspicious activity:")
    if st.button("Submit Report"):
        # Process the report (e.g., save to database or send an email)
        st.success("Thank you for your report. Your identity remains anonymous.")
        # For testing purposes, you can print the report to the console
        print("Anonymous Suspicious Activity Report:")
        print(report_text)

    # Twilio SMS Sender code
    st.sidebar.header("Twilio SMS Sender")
    twilio_account_sid = st.sidebar.text_input("Twilio Account SID", "")
    twilio_auth_token = st.sidebar.text_input("Twilio Auth Token", "")
    twilio_phone_number = st.sidebar.text_input("Twilio Phone Number", "")
    recipient_phone_number = st.sidebar.text_input("Recipient Phone Number", "")

    # (Assuming you have code to handle sending SMS via Twilio)

#<-------------------------------------------- Safety Tips Option ------------------------------------------------------>
elif option == 'Safety Tips':
    st.header("🛡️ Safety Tips and Resources")
    st.markdown("""
    - **Be Aware of Your Surroundings:** Stay alert, especially in unfamiliar areas.
    - **Avoid Isolated Places:** Stick to well-lit and populated areas, particularly at night.
    - **Secure Your Belongings:** Keep valuables out of sight to prevent theft.
    - **Trust Your Instincts:** If something feels wrong, remove yourself from the situation.
    - **Emergency Contacts:**
        - Police: **100**
        - Ambulance: **102**
        - Women's Helpline: **1091**
        - Child Helpline: **1098**
    - **Local Resources:**
        - [Delhi Police Website](https://www.delhipolice.nic.in/)
        - [Safety Apps and Tools](https://www.delhipolice.nic.in/apps.html)
    """)

# <------------------------------------------------- Model Training DNN ------------------------------------------------------>    

# Read the uploaded file
data = df.copy()

target_columns = ['murder', 'rape', 'gangrape', 'robbery', 'theft', 'assualt murders', 'sexual harassement']

feature_columns = ['lat', 'long']

# Identifying numeric columns (excluding 'nm_pol' which might be a string)
numeric_cols = [col for col in data.columns if data[col].dtype != 'object']

# Preprocessing
# Handling outliers: Remove outliers using IQR method for numeric columns
Q1 = data[numeric_cols].quantile(0.25)
Q3 = data[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data[numeric_cols] < (Q1 - 1.5 * IQR)) | (data[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Initializing separate scalers for features and target variables
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Splitting the dataset
X = data[feature_columns]  # Features
y = data[['lat', 'long'] + target_columns]   # Target variables (lat, long, and crime numbers)
X_train_dnn, X_test_dnn, y_train_dnn, y_test_dnn = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform the features
X_train_dnn = feature_scaler.fit_transform(X_train_dnn)
X_test_dnn = feature_scaler.transform(X_test_dnn)

# Fit and transform the target variables
y_train_scaled = target_scaler.fit_transform(y_train_dnn)
y_test_scaled = target_scaler.transform(y_test_dnn)

# Caching the model training to optimize memory usage
@st.cache_resource
def train_dnn_model(X_train_dnn, y_train_scaled):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(2,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(9, activation='linear')  # Update to match the number of columns in target variable
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Training the model
    model.fit(X_train_dnn, y_train_scaled, epochs=10, batch_size=32, validation_split=0.2)
    return model

model = train_dnn_model(X_train_dnn, y_train_scaled)

# Evaluating the model
y_pred_scaled = model.predict(X_test_dnn)
mse = mean_squared_error(y_test_scaled, y_pred_scaled)
rmse = mean_squared_error(y_test_scaled, y_pred_scaled, squared=False)

st.subheader("Model Evaluation Metrics")
st.write(f"Mean Squared Error: {mse}")
st.write(f"Root Mean Squared Error: {rmse}")

# Inverse transform the scaled predictions to the original scale
predictions_original_scale = pd.DataFrame(target_scaler.inverse_transform(y_pred_scaled), columns=['pred_lat', 'pred_long'] + target_columns)

# Clip negative values to 0
predictions_original_scale[target_columns] = predictions_original_scale[target_columns].clip(lower=0).astype(int)

st.write("Predictions for Number of Crimes Per Location")
st.write(predictions_original_scale)
