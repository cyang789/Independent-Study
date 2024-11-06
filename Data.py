#should first pull requests
import json
import os

# hello :)

# Define the search URL and parameters
api_url = 'https://openi.nlm.nih.gov/api/search'
params = {
    'query': 'chest xray',
    'coll': 'mpx',
    'it': 'x,xg',
    'subSetId': 'all',
    'size': 10  # Number of images to retrieve
}

# Make the request to Open-i API
response = requests.get(api_url, params=params)

# Check if request was successful
if response.status_code == 200:
    data = response.json()

    # Ensure the directory for images exists
    os.makedirs('images', exist_ok=True)

    # Loop through the results and download images
    for i, item in enumerate(data.get('list', [])):
        image_url = item['imgLarge']
        image_name = f"images/image_{i+1}.jpg"

        # Download the image
        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            with open(image_name, 'wb') as img_file:
                img_file.write(img_response.content)
            print(f"Downloaded {image_name}")
        else:
            print(f"Failed to download image {i+1}")

    # Save the metadata to a JSON file
    with open('image_metadata.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("Images and metadata downloaded successfully!")

else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
