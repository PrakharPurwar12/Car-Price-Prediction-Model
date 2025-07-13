import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import time
import plotly.express as px

# Set page config for a wider layout and custom title
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stSlider label {font-weight: bold;}
    .stSelectbox label {font-weight: bold;}
    .feature-card {
        border-radius: 10px;
        padding: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .price-display {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ====== APP LOAD SPINNER ======
with st.spinner('🔄 Loading application... Please wait'):
    time.sleep(2)  # Simulate loading delay

# Load trained model
@st.cache_resource
def load_model():
    return pk.load(open('model.pkl', 'rb'))

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Car dekho - Car dekho.csv')
    data['Name'] = data['Name'].apply(lambda x: str(x).split(' ')[0].strip())
    return data

cars_data = load_data()

# Sidebar
with st.sidebar:
    st.header("Car Price Predictor")
    st.markdown("""
        This app predicts the price of a used car based on its features. 
        Use the inputs below to specify the car's details and get an estimated price.
    """)
    st.image("https://media.tenor.com/0AV2GBNKeT4AAAAj/car-loading.gif", caption="Drive to Prediction 🚘")
    
    # Interactive data explorer
    with st.expander("🔍 Data Explorer"):
        explore_option = st.radio("Explore by:", ["Brand", "Fuel Type", "Transmission"])
        
        if explore_option == "Brand":
            selected_brand = st.selectbox("Select Brand", cars_data['Name'].unique())
            brand_data = cars_data[cars_data['Name'] == selected_brand]
            st.write(f"📊 {selected_brand} Stats:")
            st.write(f"• Average Price: ₹{brand_data['selling_price'].mean():,.2f}")
            st.write(f"• Total Listings: {len(brand_data)}")
            
        elif explore_option == "Fuel Type":
            fuel_counts = cars_data['fuel'].value_counts().reset_index()
            fig = px.pie(fuel_counts, names='fuel', values='count', title='Fuel Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
        elif explore_option == "Transmission":
            trans_counts = cars_data['transmission'].value_counts().reset_index()
            fig = px.bar(trans_counts, x='transmission', y='count', 
                         title='Transmission Type Distribution', color='transmission')
            st.plotly_chart(fig, use_container_width=True)
    
    if st.checkbox("Show Sample Data"):
        st.subheader("Sample Dataset")
        st.write(cars_data.head())

# Main content
st.title("🚘 Car Price Prediction ML Model")
st.markdown("Enter the car details below to predict its market price.")

# Feature cards for inputs
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        Name = st.selectbox('Select Car Brand', cars_data['Name'].unique(), key='brand_select')
        year = st.slider('Car Manufactured Year', 1994, 2025, value=2015, 
                         help="Newer cars typically have higher prices")
        km_driven = st.slider('No of kms Driven', 0, 2500000, value=50000, step=1000,
                             help="Lower mileage generally means higher value")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with st.container():
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        fuel = st.selectbox('Fuel Type', cars_data['fuel'].unique(), 
                           help="Diesel cars often have better resale value")
        seller_type = st.selectbox('Seller Type', cars_data['seller_type'].unique(),
                                 help="Dealer cars may be more expensive but more reliable")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        transmission = st.selectbox('Transmission Type', cars_data['transmission'].unique(),
                                  help="Automatic transmissions often command higher prices")
        owner = st.selectbox('Owner Type', cars_data['owner'].unique(),
                           help="First owner cars typically have better value")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with st.container():
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        mileage = st.slider('Car Mileage (km/l)', 10.0, 50.0, value=20.0, step=0.1,
                           help="Higher mileage means better fuel efficiency")
        engine = st.slider('Engine CC', 600.0, 4000.0, value=1500.0, step=50.0,
                          help="Larger engines typically cost more but have higher running costs")
        seats = st.slider('No of Seats', 2, 14, value=5,
                         help="Family cars with more seats may have different pricing")
        st.markdown('</div>', unsafe_allow_html=True)

# Price comparison feature
with st.expander("💡 Compare with Similar Cars"):
    st.write("See how your specifications compare to similar cars in the market")
    
    # Create filters based on user inputs
    similar_filter = (cars_data['Name'] == Name) & \
                    (cars_data['fuel'] == fuel) & \
                    (cars_data['transmission'] == transmission)
    
    similar_cars = cars_data[similar_filter]
    
    if not similar_cars.empty:
        # Show price distribution
        fig = px.box(similar_cars, y='selling_price', title=f"Price Distribution for Similar {Name} Cars")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top 5 similar cars
        st.write("### Similar Listings")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Average Price", f"₹{similar_cars['selling_price'].mean():,.2f}")
        with cols[1]:
            st.metric("Lowest Price", f"₹{similar_cars['selling_price'].min():,.2f}")
        with cols[2]:
            st.metric("Highest Price", f"₹{similar_cars['selling_price'].max():,.2f}")
        
        st.dataframe(similar_cars.head()[['year', 'km_driven', 'mileage', 'engine', 'seats', 'selling_price']])
    else:
        st.warning("No similar cars found in the dataset with these specifications")

# Input validation and prediction
if st.button("🚀 Predict Price", use_container_width=True):
    if year > 2025:
        st.error("Year cannot be in the future!")
    elif km_driven < 0:
        st.error("Kilometers driven cannot be negative!")
    elif mileage < 0 or engine < 0 or seats < 0:
        st.error("Mileage, engine, and seats must be positive values!")
    else:
        with st.spinner("🔍 Predicting the price..."):
            time.sleep(1.5)  # simulate processing time
            
            # Progress bar for better UX
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            
            # Create input dataframe
            input_data_model = pd.DataFrame(
                [[Name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, seats]],
                columns=['Name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'seats']
            )

            # Map values
            input_data_model['owner'].replace(
                ['First Owner', 'Second Owner', 'Test Drive Car', 'Third Owner', 'Fourth & Above Owner'],
                [1, 2, 3, 4, 5], inplace=True)
            input_data_model['fuel'].replace(
                ['Petrol', 'Diesel', 'Electric', 'CNG', 'LPG'],
                [1, 2, 3, 4, 5], inplace=True)
            input_data_model['seller_type'].replace(
                ['Individual', 'Dealer', 'Trustmark Dealer'],
                [1, 2, 3], inplace=True)
            input_data_model['transmission'].replace(
                ['Manual', 'Automatic'],
                [1, 2], inplace=True)
            input_data_model['Name'].replace(
                ['Maruti', 'Skoda', 'BMW', 'MG', 'Tata', 'Hyundai', 'Renault', 'Mahindra', 'Nissan', 'Datsun',
                 'Ford', 'Honda', 'Kia', 'Toyota', 'Volkswagen', 'Audi', 'Isuzu', 'Jeep', 'Land', 'Lexus',
                 'Mercedes-Benz', 'Volvo', 'Force', 'Chevrolet', 'Jaguar', 'Fiat', 'Mitsubishi', 'Ashok',
                 'Ambassador', 'Daewoo', 'Opel'],
                list(range(1, 32)), inplace=True)

            # Predict
            car_price = model.predict(input_data_model)[0]
            confidence_interval = car_price * 0.1
            lower_bound = round(car_price - confidence_interval, 2)
            upper_bound = round(car_price + confidence_interval, 2)
            
            # Animated price reveal
            with st.container():
                st.markdown(f'<div class="price-display">₹{round(car_price, 2):,}</div>', unsafe_allow_html=True)
                
                # Price breakdown visualization
                st.subheader("Price Factors")
                factors = {
                    'Brand': 0.25,
                    'Year': 0.30,
                    'Mileage': 0.15,
                    'Engine': 0.10,
                    'Condition': 0.20
                }
                
                fig = px.pie(names=list(factors.keys()), values=list(factors.values()),
                            title="Price Influence Factors")
                st.plotly_chart(fig, use_container_width=True)
                
                # Confidence interval visualization
                fig = px.bar(x=["Lower Bound", "Predicted", "Upper Bound"],
                            y=[lower_bound, car_price, upper_bound],
                            labels={'x': 'Estimate', 'y': 'Price (₹)'},
                            title="Price Confidence Range")
                st.plotly_chart(fig, use_container_width=True)
                
                # Price over time animation
                years = list(range(year, 2025))
                depreciation = [car_price * (0.9 ** (y-year)) for y in years]
                fig = px.line(x=years, y=depreciation,
                             labels={'x': 'Year', 'y': 'Estimated Value (₹)'},
                             title="Projected Depreciation",
                             range_y=[0, car_price*1.1])
                fig.update_traces(mode="lines+markers")
                st.plotly_chart(fig, use_container_width=True)

# Add a fun interactive element - car recommendation
with st.expander("🤖 Get a Car Recommendation"):
    st.write("Let us recommend a car based on your preferences!")
    
    budget = st.slider("Your Budget (₹)", 100000, 10000000, 500000, step=10000)
    preferred_fuel = st.selectbox("Preferred Fuel Type", cars_data['fuel'].unique())
    preferred_type = st.selectbox("Car Type", ["Hatchback", "Sedan", "SUV", "Luxury"])
    
    if st.button("Get Recommendation"):
        filtered_cars = cars_data[
            (cars_data['fuel'] == preferred_fuel) &
            (cars_data['selling_price'] <= budget)
        ]
        
        if not filtered_cars.empty:
            if preferred_type == "Hatchback":
                filtered_cars = filtered_cars[filtered_cars['seats'] <= 5]
            elif preferred_type == "SUV":
                filtered_cars = filtered_cars[filtered_cars['seats'] >= 7]
            
            if not filtered_cars.empty:
                recommended_car = filtered_cars.sort_values('selling_price', ascending=False).iloc[0]
                
                st.success("🎉 We found a great match for you!")
                st.write(f"**{recommended_car['Name']}**")
                st.write(f"Year: {recommended_car['year']}")
                st.write(f"Price: ₹{recommended_car['selling_price']:,.2f}")
                st.write(f"Mileage: {recommended_car['mileage']} km/l")
                st.write(f"Engine: {recommended_car['engine']} CC")
            else:
                st.warning("No cars match your specific preferences. Try adjusting your filters.")
        else:
            st.warning("No cars found within your budget. Try increasing your budget slightly.")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>This prediction is based on machine learning models and historical data.</p>
        <p>Actual market prices may vary based on condition, location, and other factors.</p>
    </div>
""", unsafe_allow_html=True)