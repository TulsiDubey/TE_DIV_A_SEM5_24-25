
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import base64

# Set the page config for better presentation
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Function to load and encode the video in base64 format
def get_base64_encoded_video(video_path):
    video_file = open(video_path, 'rb')  # Open video file in binary mode
    video_bytes = video_file.read()  # Read the video file as bytes
    base64_encoded_video = base64.b64encode(video_bytes).decode()  # Encode and decode as a string
    video_file.close()
    return base64_encoded_video

# Background Video: encode the local video file
video_base64 = get_base64_encoded_video("C:\\Users\\saifk\\OneDrive\\Desktop\\project\\bg.mp4")


# Embed the video and a dark overlay for readability
st.markdown(
    f"""
    <style>
    .stApp {{
        background: transparent;
    }}
    /* Full-screen video background */
    video {{
        position: fixed;
        right: 0;
        bottom: 0;
        min-width: 100%;
        min-height: 100%;
        z-index: -2;
    }}
    /* Dark overlay to improve contrast */
    .overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent dark overlay */
        z-index: -1;
    }}
    /* Style for the main content to make it pop */
    .main-content {{
        color: #ffffff;  /* White text for better readability */
        padding: 20px; /* Padding for spacing */
        background-color: rgba(0, 0, 0, 0.7); /* Dark background for content */
        border-radius: 15px; /* Rounded corners for a modern look */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5); /* Box shadow for a lift effect */
    }}
    /* Style for headers */
    h1, h2, h3 {{
        color: #ffffff; /* White color for headers */
    }}
    /* Customize button and input styling */
    .stTextInput, .stButton {{
        font-size: 1.2rem; /* Larger font for inputs and buttons */
        color: #ffffff;
        background-color: #ffffff; /* White background for buttons */
        border: 2px solid #ffffff;
    }}
    .stButton:hover {{
        background-color: #f0f0f0; /* Slightly darker white on hover */
    }}
    </style>

    <video autoplay muted loop>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    <div class="overlay"></div>
    """,
    unsafe_allow_html=True
)

# Main content block with improved visibility
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Price Predictor App")
st.write("""
    Predict the stock prices using historical data and advanced machine learning models.
    Use the input field to search for the stock you’re interested in.
""")

# Create two columns for better layout
col1, col2 = st.columns([1, 2])

with col1:
    # Input for Stock ID
    stock = st.text_input("Enter Stock Ticker", "GOOG", help="Enter a valid stock symbol like AAPL, MSFT, or TSLA")

    # Date range slider
    st.subheader("Select the date range for stock data")
    start_date = st.date_input("Start Date", datetime(2004, 9, 1))
    end_date = st.date_input("End Date", datetime.now())

    # Button to download data
    if st.button("Download Stock Data"):
        with st.spinner('Downloading stock data...'):
            stock_data = yf.download(stock, start=start_date, end=end_date)
            st.success(f'Data downloaded for {stock}!')
    else:
        st.warning("Please press 'Download Stock Data' to load data")

# Check if stock_data is available before proceeding
if 'stock_data' in locals():
    # Load Model
    with st.spinner('Loading the pre-trained model...'):
        model = load_model("Latest_stock_price_model.keras")
        st.success("Model loaded successfully!")

    # Display Stock Data
    st.subheader("📊 Stock Data Overview")
    st.write(stock_data)

    # Calculate Moving Averages
    splitting_len = int(len(stock_data) * 0.7)
    x_test = pd.DataFrame(stock_data.Close[splitting_len:])

    # Helper function for plotting
    def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None, title=""):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(full_data.index, full_data.Close, label="Original Data", color='blue', alpha=0.6)
        ax.plot(full_data.index, values, label=title, color='orange')
        if extra_data:
            ax.plot(full_data.index, extra_dataset, label="Extra Data", color='green', linestyle="--")
        plt.title(f"{title} and Original Close Prices", color="#ffffff")
        plt.legend()
        return fig

    # Plot Moving Averages
    st.subheader('📈 Moving Averages (MA)')
    with st.expander("Show Moving Averages Charts"):
        stock_data['MA_250'] = stock_data['Close'].rolling(250).mean()
        stock_data['MA_200'] = stock_data['Close'].rolling(200).mean()
        stock_data['MA_100'] = stock_data['Close'].rolling(100).mean()

        st.pyplot(plot_graph((15, 6), stock_data['MA_250'], stock_data, 0, title="250-Day MA"))
        st.pyplot(plot_graph((15, 6), stock_data['MA_200'], stock_data, 0, title="200-Day MA"))
        st.pyplot(plot_graph((15, 6), stock_data['MA_100'], stock_data, 0, title="100-Day MA"))
    
    # Scaling Data for Predictions
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[['Close']])

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Make Predictions
    predictions = model.predict(x_data)
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Create DataFrame for Plotting
    ploting_data = pd.DataFrame({
        'Original Test Data': inv_y_test.reshape(-1),
        'Predictions': inv_pre.reshape(-1)
    }, index=stock_data.index[splitting_len+100:])

    # Display Predictions
    st.subheader("🔮 Predictions vs Original Data")
    st.write(ploting_data)

    # Plot Predictions
    st.subheader('📉 Original Close Price vs Predicted Close Price')
    fig = plt.figure(figsize=(15, 6))
    plt.plot(pd.concat([stock_data.Close[:splitting_len+100], ploting_data], axis=0))
    plt.legend(["Data not used", "Original Test Data", "Predicted Test Data"])
    plt.title("Original vs Predicted Prices", color="#ffffff")
    st.pyplot(fig)

    # Footer
    st.write("This app is powered by Machine Learning and real-time stock data from Yahoo Finance.")
else:
    st.warning("Please download the stock data to proceed.")

# End of the main content block
st.markdown('</div>', unsafe_allow_html=True)
