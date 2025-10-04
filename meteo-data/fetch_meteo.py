import requests
from requests.auth import HTTPBasicAuth

username = "khan_fasih"
password = "89C002MTO8NyYgQpRrXh"

regions = [
    ("karachi", "pakistan", 24.8607, 67.0011),
    ("lahore", "pakistan", 31.5204, 74.3587),
    ("islamabad", "pakistan", 33.6844, 73.0479),
    ("rawalpindi", "pakistan", 33.5651, 73.0169),
    ("peshawar", "pakistan", 34.0151, 71.5249),
    ("quetta", "pakistan", 30.1798, 66.9750),
    ("multan", "pakistan", 30.1575, 71.5249),
    ("hyderabad", "pakistan", 25.3960, 68.3578),
    ("mumbai", "india", 19.0760, 72.8777),
    ("delhi", "india", 28.7041, 77.1025),
    ("kolkata", "india", 22.5726, 88.3639),
    ("chennai", "india", 13.0827, 80.2707),
    ("bengaluru", "india", 12.9716, 77.5946),
    ("hyderabad", "india", 17.3850, 78.4867),
    ("ahmedabad", "india", 23.0225, 72.5714),
    ("pune", "india", 18.5204, 73.8567),
    ("surat", "india", 21.1702, 72.8311),
    ("jaipur", "india", 26.9124, 75.7873),
    ("lucknow", "india", 26.8467, 80.9462),
    ("patna", "india", 25.5941, 85.1376),
    ("dhaka", "bangladesh", 23.8103, 90.4125),
    ("chittagong", "bangladesh", 22.3569, 91.7832),
    ("colombo", "sri lanka", 6.9271, 79.8612),
    ("kandy", "sri lanka", 7.2906, 80.6337),
    ("kathmandu", "nepal", 27.7172, 85.3240),
]

url = (
    "https://api.meteomatics.com/"
    "2025-01-01T00:00:00Z--2025-12-31T00:00:00Z:PT24H/"
    "heavy_rain_warning_24h:idx/"
    "24.941665,67.1246654/csv?source=mix"
)

print(url)

# response = requests.get(url, auth=HTTPBasicAuth(username, password))

# if response.status_code == 200:
#     with open("meteomatics_data.csv", "wb") as f:
#         f.write(response.content)
#     print("CSV file saved as meteomatics_data.csv")
# else:
#     print(f"Error: {response.status_code} - {response.text}")

