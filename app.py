# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import json
from datetime import datetime
import joblib

# IMPORTANT: Configure Flask to serve static files from 'templates/static'
app = Flask(__name__, template_folder='templates', static_folder='templates/static')

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DATA_FILE'] = 'crop_production.csv'
app.config['EDA_STATIC_IMAGE_FOLDER'] = 'eda_images' # Subfolder within static for pre-generated images

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Ensure the static image folder exists (though it should if you placed images there)
os.makedirs(os.path.join(app.static_folder, app.config['EDA_STATIC_IMAGE_FOLDER']), exist_ok=True)


# Load your CatBoost model and feature columns
try:
    model = joblib.load('CatBoost_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    print("Model and feature columns loaded successfully")
except Exception as e:
    print(f"Error loading model or feature columns: {e}")
    model = None
    feature_columns = None

# States and districts dictionary (for dropdowns)
statesAndDistricts = {
    "Andhra Pradesh": ["Anantapur", "Chittoor", "East Godavari", "Guntur", "Kadapa", "Krishna", "Kurnool", "Nellore", "Prakasam", "Srikakulam", "Visakhapatnam", "Vizianagaram", "West Godavari"],
    "Arunachal Pradesh": ["Tawang", "West Kameng", "East Kameng", "Papum Pare", "Kurung Kumey", "Kra Daadi", "Lower Subansiri", "Upper Subansiri", "West Siang", "East Siang", "Siang", "Upper Siang", "Lower Dibang Valley", "Dibang Valley", "Anjaw", "Lohit", "Namsai", "Changlang", "Tirap", "Longding"],
    "Assam": ["Baksa", "Barpeta", "Biswanath", "Bongaigaon", "Cachar", "Charaideo", "Chirang", "Darrang", "Dhemaji", "Dhubri", "Dibrugarh", "Goalpara", "Golaghat", "Hailakandi", "Hojai", "Jorhat", "Kamrup Metropolitan", "Kamrup", "Karbi Anglong", "Karimganj", "Kokrajhar", "Lakhimpur", "Majuli", "Morigaon", "Nagaon", "Nalbari", "Sivasagar", "Sonitpur", "South Salmara-Mankachar", "Tinsukia", "Udalguri", "West Karbi Anglong"],
    "Bihar": ["Araria", "Arwal", "Aurangabad", "Banka", "Begusarai", "Bhagalpur", "Bhojpur", "Buxar", "Darbhanga", "East Champaran", "Gaya", "Gopalganj", "Jamui", "Jehanabad", "Kaimur", "Katihar", "Khagaria", "Kishanganj", "Lakhisarai", "Madhepura", "Madhubani", "Munger", "Muzaffarpur", "Nalanda", "Nawada", "Patna", "Purnia", "Rohtas", "Saharsa", "Samastipur", "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", "Siwan", "Supaul", "Vaishali", "West Champaran"],
    "Chhattisgarh": ["Balod", "Baloda Bazar", "Balrampur", "Bastar", "Bemetara", "Bijapur", "Bilaspur", "Dantewada", "Dhamtari", "Durg", "Gariaband", "Janjgir-Champa", "Jashpur", "Kanker", "Kabirdham", "Kondagaon", "Korba", "Korea", "Mahasamund", "Mungeli", "Narayanpur", "Raigarh", "Rajnandgaon", "Sukma", "Surajpur", "Surguja"],
    "Goa": ["North Goa", "South Goa"],
    "Gujarat": ["Ahmedabad", "Amreli", "Anand", "Aravalli", "Banaskantha", "Bharuch", "Bhavnagar", "Botad", "Chhota Udaipur", "Dahod", "Dang", "Devbhoomi Dwarka", "Gandhinagar", "Gir Somnath", "Jamnagar", "Junagadh", "Kheda", "Kutch", "Mahisagar", "Mehsana", "Morbi", "Narmada", "Navsari", "Panchmahal", "Patan", "Porbandar", "Rajkot", "Sabarkantha", "Surat", "Surendranagar", "Tapi", "Vadodara", "Valsad"],
    "Haryana": ["Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad", "Gurugram", "Hisar", "Jhajjar", "Jind", "Kaithal", "Karnal", "Kurukshetra", "Mahendragarh", "Nuh", "Palwal", "Panchkula", "Panipat", "Rewari", "Rohtak", "Sirsa", "Sonipat", "Yamunanagar"],
    "Himachal Pradesh": ["Bilaspur", "Chamba", "Hamirpur", "Kangra", "Kinnaur", "Kullu", "Lahaul and Spiti", "Mandi", "Shimla", "Sirmaur", "Solan", "Una"],
    "Jharkhand": ["Bokaro", "Chatra", "Deoghar", "Dhanbad", "Dumka", "East Singhbhum", "Garhwa", "Giridih", "Godda", "Gumla", "Hazaribagh", "Jamtara", "Khunti", "Koderma", "Latehar", "Lohardaga", "Pakur", "Palamu", "Ramgarh", "Ranchi", "Sahibganj", "Saraikela-Kharsawan", "Simdega", "West Singhbhum"],
    "Karnataka": ["Bagalkot", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban", "Bidar", "Chamarajanagar", "Chikballapur", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada", "Davangere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi", "Kodagu", "Kolar", "Koppal", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumakuru", "Udupi", "Uttara Kannada", "Vijayanagara", "Yadgir"],
    "Kerala": ["Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod", "Kollam", "Kottayam", "Kozhikode", "Malappuram", "Palakkad", "Pathanamthitta", "Thiruvananthapuram", "Thrissur", "Wayanad"],
    "Madhya Pradesh": ["Agar Malwa", "Alirajpur", "Anuppur", "Ashok Nagar", "Balaghat", "Barwani", "Betul", "Bhind", "Bhopal", "Burhanpur", "Chhatarpur", "Chhindwara", "Damoh", "Datia", "Dewas", "Dhar", "Dindori", "Guna", "Gwalior", "Harda", "Hoshangabad", "Indore", "Jabalpur", "Jhabua", "Katni", "Khandwa", "Khargone", "Mandla", "Mandsaur", "Morena", "Narsinghpur", "Neemuch", "Panna", "Raisen", "Rajgarh", "Ratlam", "Rewa", "Sagar", "Satna", "Sehore", "Seoni", "Shahdol", "Shajapur", "Sheopur", "Shivpuri", "Sidhi", "Singrauli", "Tikamgarh", "Ujjain", "Umaria", "Vidisha"],
    "Maharashtra": ["Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara", "Buldhana", "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli", "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad", "Palghar", "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal"],
    "Manipur": ["Bishnupur", "Chandel", "Churachandpur", "Imphal East", "Imphal West", "Jiribam", "Kakching", "Kamjong", "Kangpokpi", "Noney", "Pherzawl", "Senapati", "Tamenglong", "Tengnoupal", "Thoubal", "Ukhrul"],
    "Meghalaya": ["East Garo Hills", "East Jaintia Hills", "East Khasi Hills", "North Garo Hills", "Ri Bhoi", "South Garo Hills", "South West Garo Hills", "South West Khasi Hills", "West Garo Hills", "West Jaintia Hills", "West Khasi Hills"],
    "Mizoram": ["Aizawl", "Champhai", "Kolasib", "Lawngtlai", "Lunglei", "Mamit", "Saiha", "Serchhip"],
    "Nagaland": ["Dimapur", "Kiphire", "Kohima", "Longleng", "Mokokchung", "Mon", "Peren", "Phek", "Tuensang", "Wokha", "Zunheboto"],
    "Odisha": ["Angul", "Balangir", "Balasore", "Bargarh", "Bhadrak", "Boudh", "Cuttack", "Deogarh", "Dhenkanal", "Gajapati", "Ganjam", "Jagatsinghpur", "Jajpur", "Jharsuguda", "Kalahandi", "Kandhamal", "Kendrapara", "Kendujhar", "Khordha", "Koraput", "Malkangiri", "Mayurbhanj", "Nabarangpur", "Nayagarh", "Nuapada", "Puri", "Rayagada", "Sambalpur", "Sonepur", "Sundergarh"],
    "Punjab": ["Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib", "Fazilka", "Firozpur", "Gurdaspur", "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana", "Mansa", "Moga", "Muktsar", "Nawanshahr", "Pathankot", "Patiala", "Rupnagar", "Sahibzada Ajit Singh Nagar", "Sangrur", "Tarn Taran"],
    "Rajasthan": ["Ajmer", "Alwar", "Banswara", "Baran", "Barmer", "Bharatpur", "Bhilwara", "Bikaner", "Bundi", "Chittorgarh", "Churu", "Dausa", "Dholpur", "Dungarpur", "Ganganagar", "Hanumangarh", "Jaipur", "Jaisalmer", "Jalore", "Jhalawar", "Jhunjhunu", "Jodhpur", "Karauli", "Kota", "Nagaur", "Pali", "Pratapgarh", "Rajsamand", "Sawai Madhopur", "Sikar", "Sirohi", "Tonk", "Udaipur"],
    "Sikkim": ["East Sikkim", "North Sikkim", "South Sikkim", "West Sikkim"],
    "Tamil Nadu": ["Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore", "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kancheepuram", "Karur", "Krishnagiri", "Madurai", "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai", "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi", "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli", "Tiruppur", "Tiruvallur", "Tiruvarur", "Vellore", "Viluppuram", "Virudhunagar"],
    "Telangana": ["Adilabad", "Bhadradri Kothagudem", "Hyderabad", "Jagtial", "Jangaon", "Jayashankar Bhoopalpally", "Jogulamba Gadwal", "Kamareddy", "Karimnagar", "Khammam", "Komaram Bheem Asifabad", "Mahabubabad", "Mahabubnagar", "Mancherial", "Medak", "Medchal Malkajgiri", "Mulugu", "Nagarkurnool", "Nalgonda", "Narayanpet", "Nirmal", "Nizamabad", "Peddapalli", "Rajanna Sircilla", "Rangareddy", "Sangareddy", "Siddipet", "Suryapet", "Vikarabad", "Wanaparthy", "Warangal (Rural)", "Warangal (Urban)", "Yadadri Bhuvanagiri"],
    "Tripura": ["Dhalai", "Gomati", "Khowai", "North Tripura", "Sepahijala", "South Tripura", "Unakoti", "West Tripura"],
    "Uttar Pradesh": ["Agra", "Aligarh", "Allahabad (Prayagraj)", "Ambedkar Nagar", "Amethi", "Amroha", "Auraiya", "Azamgarh", "Baghpat", "Bahraich", "Ballia", "Balrampur", "Banda", "Barabanki", "Bareilly", "Basti", "Bhadohi", "Bijnor", "Budaun", "Bulandshahr", "Chandauli", "Chitrakoot", "Deoria", "Etah", "Etawah", "Farrukhabad", "Fatehpur", "Firozabad", "Gautam Buddha Nagar", "Ghaziabad", "Ghazipur", "Gonda", "Gorakhpur", "Hamirpur", "Hapur", "Hardoi", "Hathras", "Jalaun", "Jaunpur", "Jhansi", "Kannauj", "Kanpur Dehat", "Kanpur Nagar", "Kasganj", "Kaushambi", "Kushinagar", "Lakhimpur Kheri", "Lalitpur", "Lucknow", "Maharajganj", "Mahoba", "Mainpuri", "Mathura", "Mau", "Meerut", "Mirzapur", "Moradabad", "Muzaffarnagar", "Pilibhit", "Pratapgarh", "Raebareli", "Rampur", "Saharanpur", "Sambhal", "Sant Kabir Nagar", "Shahjahanpur", "Shamli", "Shravasti", "Siddharth Nagar", "Sitapur", "Sonbhadra", "Sultanpur", "Unnao", "Varanasi"],
    "Uttarakhand": ["Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun", "Haridwar", "Nainital", "Pauri Garhwal", "Pithoragarh", "Rudraprayag", "Tehri Garhwal", "Udham Singh Nagar", "Uttarkashi"],
    "West Bengal": ["Alipurduar", "Bankura", "Birbhum", "Cooch Behar", "Dakshin Dinajpur", "Darjeeling", "Hooghly", "Howrah", "Jalpaiguri", "Jhargram", "Kalimpong", "Kolkata", "Malda", "Mursridabad", "Nadia", "North 24 Parganas", "Paschim Bardhaman", "Paschim Medinipur", "Purba Bardhaman", "Purba Medinipur", "Purulia", "South 24 Parganas", "Uttar Dinajpur"]
}

# Load your dataset for EDA
def load_crop_data():
    """Load the crop production dataset"""
    try:
        df = pd.read_csv(app.config['DATA_FILE'])
        return df
    except FileNotFoundError:
        print(f"Warning: {app.config['DATA_FILE']} not found. Creating sample data.")
        return create_sample_data()

def create_sample_data():
    """Create sample data if CSV file is not found"""
    np.random.seed(42)
    data = {
        'Year': np.random.randint(2000, 2024, 1000),
        'State': np.random.choice(['Punjab', 'Haryana', 'Uttar Pradesh', 'Maharashtra', 'Karnataka'], 1000),
        'District': np.random.choice(['District_A', 'District_B', 'District_C', 'District_D'], 1000),
        'Crop': np.random.choice(['Wheat', 'Rice', 'Corn', 'Soybeans', 'Cotton'], 1000),
        'Season': np.random.choice(['Kharif', 'Rabi', 'Zaid'], 1000),
        'Area': np.random.uniform(100, 5000, 1000),
        'Production': np.random.uniform(1000, 25000, 1000),
        'Yield': np.random.uniform(1.5, 4.5, 1000)
    }
    df = pd.DataFrame(data)
    df['Yield'] = df['Production'] / df['Area']
    return df

# EDA Functions (These will generate base64 images for uploaded CSVs)
# IMPORTANT: These functions should be robust to missing columns and handle errors gracefully
def generate_top5_crops(df):
    try:
        plt.figure(figsize=(12, 8))
        if 'Crop' not in df.columns or 'Production' not in df.columns:
            raise ValueError("Missing 'Crop' or 'Production' columns for Top 5 Crops visualization.")
        
        top_crops = df.groupby('Crop')['Production'].sum().nlargest(5)
        plt.bar(top_crops.index, top_crops.values)
        plt.title('Top 5 Crops by Production', fontsize=16, fontweight='bold')
        plt.xlabel('Crop Type', fontsize=12)
        plt.ylabel('Total Production', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Error in generate_top5_crops: {e}")
        return None

def generate_top10_states(df):
    try:
        plt.figure(figsize=(14, 8))
        if 'State' not in df.columns or 'Production' not in df.columns:
            raise ValueError("Missing 'State' or 'Production' columns for Top 10 States visualization.")
        
        top_states = df.groupby('State')['Production'].sum().nlargest(10)
        plt.barh(top_states.index, top_states.values)
        plt.title('Top 10 States by Production', fontsize=16, fontweight='bold')
        plt.xlabel('Total Production', fontsize=12)
        plt.ylabel('State', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Error in generate_top10_states: {e}")
        return None

def generate_yield_distribution(df):
    try:
        plt.figure(figsize=(12, 8))
        if 'Yield' not in df.columns:
            if 'Production' in df.columns and 'Area' in df.columns and not df['Area'].eq(0).any():
                df['Yield'] = df['Production'] / df['Area']
            else:
                raise ValueError("Missing 'Yield' (or 'Production' and non-zero 'Area') columns for Yield Distribution visualization.")
        
        plt.hist(df['Yield'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Yield Distribution (Tonnes/Hectare)', fontsize=16, fontweight='bold')
        plt.xlabel('Yield (Tonnes/Hectare)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Error in generate_yield_distribution: {e}")
        return None

def generate_area_vs_production(df):
    try:
        plt.figure(figsize=(12, 8))
        if 'Crop' not in df.columns or 'Area' not in df.columns or 'Production' not in df.columns:
            raise ValueError("Missing 'Crop', 'Area', or 'Production' columns for Area vs Production visualization.")
        
        crop_stats = df.groupby('Crop').agg({'Area': 'mean', 'Production': 'mean'})
        plt.scatter(crop_stats['Area'], crop_stats['Production'], s=100, alpha=0.7)
        
        for i, crop in enumerate(crop_stats.index):
            plt.annotate(crop, (crop_stats['Area'].iloc[i], crop_stats['Production'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title('Area vs Production by Crop', fontsize=16, fontweight='bold')
        plt.xlabel('Average Area (Hectares)', fontsize=12)
        plt.ylabel('Average Production (Tonnes)', fontsize=12)
        plt.grid(alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Error in generate_area_vs_production: {e}")
        return None

def generate_production_by_season(df):
    try:
        plt.figure(figsize=(12, 8))
        if 'Season' not in df.columns or 'Production' not in df.columns:
            raise ValueError("Missing 'Season' or 'Production' columns for Production by Season visualization.")
        
        season_production = df.groupby('Season')['Production'].sum()
        plt.pie(season_production.values, labels=season_production.index, autopct='%1.1f%%')
        plt.title('Production Distribution by Season', fontsize=16, fontweight='bold')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Error in generate_production_by_season: {e}")
        return None

def generate_production_trends(df):
    try:
        plt.figure(figsize=(14, 8))
        if 'Year' not in df.columns or 'Production' not in df.columns:
            raise ValueError("Missing 'Year' or 'Production' columns for Production Trends visualization.")
        
        yearly_production = df.groupby('Year')['Production'].sum()
        plt.plot(yearly_production.index, yearly_production.values, marker='o', linewidth=2)
        plt.title('Total Production Over Years', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Total Production (Tonnes)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.xticks(yearly_production.index, rotation=45)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Error in generate_production_trends: {e}")
        return None

def generate_correlation_heatmap(df):
    try:
        plt.figure(figsize=(12, 8))
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or len(numeric_df.columns) < 2:
            raise ValueError("Not enough numerical columns for Correlation Heatmap visualization.")
        
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap (Numerical Features)', fontsize=16, fontweight='bold')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_base64
    except Exception as e:
        print(f"Error in generate_correlation_heatmap: {e}")
        return None

# Routes
@app.route('/')
def index():
    return render_template('index.html', statesAndDistricts=statesAndDistricts)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html', statesAndDistricts=statesAndDistricts)

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# New prediction route - handles form submission

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', statesAndDistricts=statesAndDistricts)
    try:
        if 'csv_file' in request.files and request.files['csv_file'].filename != '':
            # CSV file uploaded
            csv_file = request.files['csv_file']
            input_df = pd.read_csv(csv_file)

            # Validate required columns in CSV
            required_columns = ['State', 'District', 'Crop', 'Year', 'Season', 'Area (ha)', 
                               'Temperature (°C)', 'Rainfall (mm)', 'Humidity (%)', 
                               'Nitrogen (kg/ha)', 'Phosphorus (kg/ha)', 'Potassium (kg/ha)']
            if not all(col in input_df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in input_df.columns]
                return render_template('result.html', error=f"Missing columns in CSV: {', '.join(missing_cols)}")

            # One-hot encode categorical variables
            input_encoded = pd.get_dummies(input_df)

            # Ensure all feature columns exist and in the correct order
            for col in feature_columns:
                if col not in input_encoded:
                    input_encoded[col] = 0
            input_encoded = input_encoded[feature_columns]

            # Convert all columns to float to avoid type mismatch
            input_encoded = input_encoded.astype(float)

            # Predict for all rows
            predictions = model.predict(input_encoded)
            predictions = np.maximum(predictions, 0)  # Ensure predictions are not negative

            # Add predictions to the original DataFrame
            input_df['Predicted Production (tonnes)'] = np.round(predictions, 2)

            return render_template('result.html', predictions_df=input_df.to_html(classes='prediction-table', index=False))

        else:
            # Manual form input
            state = request.form['state']
            district = request.form['district']
            crop = request.form['crop'].title()  # Normalize to title case
            crop_year = int(request.form['crop_year'])
            season = request.form['season']
            area = float(request.form['area'])
            temperature = float(request.form['temperature'])
            rainfall = float(request.form['rainfall'])
            humidity = float(request.form['humidity'])
            nitrogen = float(request.form['nitrogen'])
            phosphorus = float(request.form['phosphorus'])
            potassium = float(request.form['potassium'])

            # Prepare DataFrame with correct column names
            input_df = pd.DataFrame({
                'State': [state],
                'District': [district],
                'Crop': [crop],
                'Year': [crop_year],
                'Season': [season],
                'Area (ha)': [area],
                'Temperature (°C)': [temperature],
                'Rainfall (mm)': [rainfall],
                'Humidity (%)': [humidity],
                'Nitrogen (kg/ha)': [nitrogen],
                'Phosphorus (kg/ha)': [phosphorus],
                'Potassium (kg/ha)': [potassium]
            })

            # One-hot encode categorical variables
            input_encoded = pd.get_dummies(input_df)

            # Ensure all feature columns exist and in the correct order
            for col in feature_columns:
                if col not in input_encoded:
                    input_encoded[col] = 0
            input_encoded = input_encoded[feature_columns]

            # Convert all columns to float to avoid type mismatch
            input_encoded = input_encoded.astype(float)

            # Predict
            prediction = max(model.predict(input_encoded)[0], 0)

            return render_template('result.html', prediction=round(prediction, 2))

    except Exception as e:
        return render_template('result.html', error=str(e))

# Serve static files from templates directory (for CSS/JS)
@app.route('/serve_css/<path:filename>')
def serve_css(filename):
    return send_from_directory('templates', filename)

@app.route('/serve_js/<path:filename>')
def serve_js(filename):
    return send_from_directory('templates', filename)

# API endpoint for pre-generated EDA images (serves static files)
@app.route('/api/eda/<visualization_type>')
def get_eda_visualization(visualization_type):
    image_filename = f"{visualization_type}.png"
    # Use url_for('static', filename=...) to get the URL for the static file
    # The 'static' endpoint is automatically created by Flask when static_folder is defined
    image_url = url_for('static', filename=f"{app.config['EDA_STATIC_IMAGE_FOLDER']}/{image_filename}")
    
    # Check if the file actually exists before returning the URL
    # This prevents a broken image link if the file is missing
    full_path = os.path.join(app.static_folder, app.config['EDA_STATIC_IMAGE_FOLDER'], image_filename)
    if os.path.exists(full_path):
        return jsonify({'image': image_url})
    else:
        return jsonify({'error': f'Image not found for {visualization_type}. Please ensure {image_filename} exists in templates/static/eda_images/'}), 404

# CSV upload and analysis endpoint for EDA (generates base64 images on the fly)
@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    if 'csv_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['csv_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            uploaded_df = pd.read_csv(io.BytesIO(file.read())) # Read directly from BytesIO
            
            visualizations = {}
            # Define the visualization functions that can be generated
            visualization_functions = {
                'top5_crops': generate_top5_crops,
                'top10_states': generate_top10_states,
                'yield_distribution': generate_yield_distribution,
                'area_vs_production': generate_area_vs_production,
                'production_by_season': generate_production_by_season,
                'production_trends': generate_production_trends,
                'correlation_heatmap': generate_correlation_heatmap
            }
            
            # Generate all possible visualizations and store their base64 strings
            for viz_name, viz_func in visualization_functions.items():
                try:
                    img_base64 = viz_func(uploaded_df)
                    if img_base64 is not None:
                        visualizations[viz_name] = f'data:image/png;base64,{img_base64}'
                    else:
                        visualizations[viz_name] = None
                except Exception as e:
                    # If a visualization fails (e.g., missing columns), store None and print error
                    visualizations[viz_name] = None 
                    print(f"Error generating {viz_name} for uploaded CSV: {e}")
            
            return jsonify({
                'success': True,
                'filename': file.filename,
                'visualizations': visualizations, # Send all generated base64 images
                'data_preview': uploaded_df.head().to_dict()
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing CSV: {str(e)}'}), 400
    
    return jsonify({'error': 'Invalid file type. Please upload a CSV file'}), 400

if __name__ == '__main__':
    app.run(debug=True)


    