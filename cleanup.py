import pandas as pd
import pydantic
from pydantic import BaseModel, ValidationError
from pathlib import Path
from typing import Optional, List
from distilabel.llms import vLLM, OpenAILLM
import json
import re

# Define the schema for the cleaned Work Order Internal Note
class CleanedNote(BaseModel):
    cleaned_text: str
    author: str
    date: str

# Define the schema for the cleaned Brand and Model
class CleanedBrandModel(BaseModel):
    brand: str
    model: str

# Function to clean the note using regex as a preliminary step
def preprocess_text(text: str) -> str:
    """
    Remove lines that start with a name and date from the Work Order Internal Note.
    """
    # Example patterns:
    # (2021-11-15) Killian
    # Killian McAteer (7/12/2021 17:31:16)
    # Rémy Dunoyer (18/12/2021 18:16:23)
    pattern = r"^(?:\(\d{4}-\d{2}-\d{2}\)\s+\w+|\w+\s+\w+\s+\(\d{1,2}/\d{1,2}/\d{4}\s+\d{2}:\d{2}:\d{2}\))\s*"
    cleaned = re.sub(pattern, '', text, flags=re.MULTILINE)
    return cleaned.strip()

# Function to remove prefixes from Brand and Model
def remove_brand_prefix(text: str, prefixes: List[str]) -> str:
    """
    Remove any specified prefixes from the Brand and Model text.
    """
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix):].strip()
    return text

# Function to remove descriptors from Brand and Model
def remove_brand_descriptors(text: str, descriptors: List[str]) -> str:
    """
    Remove descriptors such as color names and other non-essential terms from the Brand and Model text.
    """
    # Create a regex pattern to match whole words in descriptors list
    pattern = r'\b(' + '|'.join(map(re.escape, descriptors)) + r')\b'
    # Remove matched descriptors
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Remove extra spaces and unwanted characters
    cleaned_text = re.sub(r'[\-\_]', ' ', cleaned_text)  # Replace - and _ with space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Function to extract brand and model using LLM
def extract_brand_model_llm(text: str, llm: OpenAILLM) -> Optional[dict]:
    """
    Use LLM to extract brand and model from the cleaned Brand and Model text.
    """
    if pd.isna(text):
        return {"brand": "", "model": ""}
    
    # Prepare the prompt for the LLM
    prompt = (
        "Extract the brand and model from the following text. Ensure no extraneous punctuation, capitalization, whitespace, or special characters are included. Remove all trailing and leading punctuation from the extracted brand and model.\n\n"
        "Return the result in JSON format with 'brand' and 'model' fields. "
        "If the brand or model is unknown, See if a known brand below matches, and use that as the value.\n\n"
        """
        Known brands:
        ```
        AER
        Apollo
        Aprilia
        Bird
        Blog
        Bronco
        Carrera
        Currus
        Dualtron
        Ducati
        E-Twow
        EMove
        Evo Powerboards
        Fluid Freeride
        Glion
        Gotrax
        Hiboy
        Inmotion
        Inokim
        Jeep
        Kaabo
        Kugoo
        Levy
        Megawheels
        Mercane
        Nami
        Nanrobot
        Ninebot
        NIU
        NIU
        Pure Air
        Razor
        Reid
        Riley
        Segway
        Solar
        Swagtron
        Turboant
        Turbowheel
        Unagi
        Uncategorised-Scooters
        Urban Drift
        Uscooters
        Varla
        VSETT
        Weped
        Xiaomi
        ZERO
        3T Cycling
        A-bike
        Abici
        Adler
        AIST
        ALAN
        Al Carter
        Alcyon
        Alldays & Onions
        American Bicycle Company
        American Bicycle Group
        American Eagle
        American Machine and Foundry
        American Star Bicycle
        Aprilia
        Argon 18
        Ariel
        Atala
        Author
        Avanti
        Baltik vairas
        Bacchetta
        Basso_Bikes
        Batavus
        Battaglin
        Berlin & Racycle Manufacturing Company
        BH
        Bianchi
        Bickerton
        Bike Friday
        Bilenky
        Biomega
        Birdy
        BMC
        Boardman Bikes
        Bohemian Bicycles
        Bontrager
        Bootie
        Bottecchia
        Bradbury
        Brasil & Movimento
        Brennabor
        Bridgestone
        British Eagle
        Brodie Bicycles
        Brompton Bicycle
        Brunswick
        BSA
        B’Twin
        Burley Design
        Calcott Brothers
        Calfee Design
        Caloi
        Campion Cycle Company
        Cannondale
        Canyon bicycles
        Catrike
        CCM
        Centurion
        Cervélo
        Chater-Lea
        Chicago Bicycle Company
        Cilo
        Cinelli
        Citizen Bike
        Clark-Kent
        Claud Butler
        Clément
        Co-Motion Cycles
        Coker
        Colnago
        Columbia Bicycles
        Corima
        Cortina Cycles
        Coventry-Eagle- UK (defunct
        Cruzbike
        Cube
        Currys
        Cycle Force Group
        Cycles Devinci
        Cycleuropa Group
        Cyfac
        Dahon
        Dawes Cycles
        Decauville, France (defunct)
        Defiance Cycle Company
        Demorest
        Den Beste Sykkel Better known as DBS
        Derby Cycle
        De Rosa
        Cycles Devinci
        Di Blasi Industriale
        Diamant
        Diamondback Bicycles
        Dolan Bikes
        Dorel Sports
        Dunelt
        Dynacraft
        Eagle Bicycle Manufacturing Company
        Eddy Merckx Cycles
        Electra Bicycle Company
        Ellis Briggs
        Ellsworth Handcrafted Bicycles
        Emilio Bozzi
        Ērenpreiss Bicycles
        Excelsior
        Falcon Cycles
        Fat City Cycles
        Favorit
        Felt
        Flying Pigeon
        Flying Scot
        Focus Bikes
        Cycles Follis
        Folmer & Schwing
        Fondriest
        Fram
        Freddie Grubb
        Fuji Bikes
        Fyxation
        Gazelle
        Gendron Bicycles
        Genesis
        Gepida
        Ghost
        Giant Manufacturing
        Gimson
        Gitane
        Gladiator Cycle Company
        Gnome et Rhône
        Gocycle
        Gormully & Jeffery
        Gräf & Stift
        GT Bicycles
        Guerciotti
        Gustavs Ērenpreis Bicycle Factory
        Gunnar
        Halfords
        Harley-Davidson
        Haro Bikes
        Harry Quinn
        Hase bikes
        Heinkel
        Helkama
        Henley Bicycle Works
        Hercules
        Hercules
        Hero Cycles Ltd
        René Herse
        Hetchins
        Hillman
        Hoffman BMX Bikes
        Hoffmann
        Holdsworth
        Huffy
        Humber
        Hurtu
        Husqvarna
        Hutch BMX BMX Bicycle manufacturer USA
        Ibis
        Ideal Bikes
        Indian
        IFA
        Independent Fabrication
        Inspired Cycle Engineering (ICE)
        Iride
        Iron Horse Bicycles
        Islabikes – UK
        Italvega
        Ivel Cycle Works
        Iver Johnson
        Iverson
        Jan Janssen
        JMC Bicycles
        Jamis Bicycles- USA
        Kalkhoff
        Kangaroo
        Karbon Kinetics Limited
        K2 Sports
        Kent
        Kestrel USA
        Kettler
        KHS
        Kia
        Kinesis Industry
        Klein
        KOGA (formerly Koga Miyata)
        Kogswell Cycles
        Kona
        Kronan
        Kross
        KTM
        Kuota
        Kuwahara
        Laurin & Klement
        Lapierre
        LeMond
        Alexander Leutner & Co. — Russia (defunct)
        Lightning Cycle Dynamics
        Litespeed
        Look
        Louison Bobet
        Lotus, USA (defunct)
        Magna
        Malvern Star
        Marin Bikes
        Masi Bicycles
        Matchless
        Matra
        Melon Bicycles
        Mercian Cycles
        Merida Bikes
        Merlin
        Merckx
        Milwaukee Bicycle Co.
        Minerva
        Miyata
        Mochet
        Monark
        Mondia
        Mongoose
        Montague
        Moots Cycles
        Motobécane
        Moulton
        Mountain Equipment Co-op
        Murray
        Muddy Fox
        Nagasawa
        National
        Neil Pryde
        Neobike
        NEXT
        Nishiki
        Norco
        Norman Cycles
        Novara
        NSU
        Nymanbolagen
        Olmo
        Opel
        Orbea
        Órbita
        Orient Bikes
        Overman Wheel Company
        Pacific Cycle
        Pacific Cycles
        Panasonic
        Pashley Cycles
        Pedersen bicycle
        Pegas
        Peugeot
        Phillips Cycles
        Phoenix
        Pierce Cycle Company
        Pinarello
        Planet X Bikes
        Pocket Bicycles
        Pogliaghi
        Polygon Bikes
        Pope Manufacturing Company
        Premier
        Prophete
        Puch
        Quadrant Cycle Company
        Quality Bicycle Products
        Quintana Roo
        R+E Cycles
        Radio Flyer
        Rabasa Cycles
        Raleigh
        Rambler
        Rans Designs
        Razor
        Redline bicycles
        Rhoades Car
        Ridgeback
        Ridley
        Riese und Müller
        RIH
        Riley Cycle Company
        Rivendell Bicycle Works
        Roadmaster
        Roberts Cycles
        Robin Hood
        Rocky Mountain Bicycles
        ROMET Bike Factory
        ROSE Bikes
        Ross
        Rover Company
        Rowbike
        Rudge-Whitworth
        Salcano
        Samchuly
        Santa Cruz Bikes
        Santana Cycles
        Saracen Cycles
        Maskinfabriks-aktiebolaget Scania
        Schwinn Bicycle Company
        SCOTT Sports
        SE Racing now SE Bikes PK Ripper and Floval Flyer maker, USA
        Serotta
        Seven Cycles
        Shelby Cycle Company
        Shimano
        Simpel
        Simson
        Sinclair Research
        Singer
        Softride- USA
        Sohrab
        Solé Bicycle Co.
        Solex
        Solifer
        SOMA Fabrications
        Somec
        Spacelander Bicycle
        Spalding
        Sparta B.V.
        Specialized
        Speedwell bicycles
        Star Cycle Company
        Stearns
        Stelber Cycle Corp
        Stella
        Sterling Bicycle Co.
        Steyr
        Strida
        Sun Cycle & Fittings Co.
        Sunbeam
        Surly Bikes
        Suzuki
        Swift Folder
        Swing Bike
        Škoda
        Tern
        TerraTrike
        Terrot
        Thomas
        Time
        Titus
        Tommaso bikes
        Torker
        Trek Bicycle Corporation
        Trident Trikes
        Trinx
        Triumph Cycle
        Triumph (TWN)
        Tube Investments
        Tunturi
        Turner Suspension Bicycles
        Univega
        Urago
        Van Dessel Sports
        VanMoof
        Velocite Bikes
        Velomotors
        VéloSoleX
        Victoria
        Villiger
        Villy Customs
        Vindec
        VinFast
        Vitus
        Volae
        Volagi
        Wanderer
        Waverley
        Waterford Precision Cycles
        Western Wheel Works
        Whippet
        Wilderness Trail Bikes
        Wilier Triestina
        Witcomb Cycles
        Wittson Custom Ti Cycles
        Worksman Cycles
        Wright Cycle Company
        Whyte
        Xootr
        Yamaguchi Bicycles
        Yamaha
        Yeti Cycles
        YT Industries
        Zigo

        AddMotor
        AmegoEV
        Ancheer
        Ariel
        Atala
        Audi
        Aventon
        Babboe
        Bagi
        Bakcou
        Belize
        Benelli
        Benno
        BESV
        BH
        Bianchi
        Big Cat
        Biktrix
        Biomega
        Biria
        Bixs
        Blix
        BMC
        BMEBIKES
        BMW
        Bolton
        Boreal
        BPM
        Brompton
        Bulls
        Butchers
        Buzz
        Cannondale
        CERO
        Christini
        Civia
        Coboc
        Colnago
        Corratec
        CUBE
        Day6
        Daymak
        DCO
        Del Sol
        Delfast
        Desiknio
        Devinci
        Diamondback
        Diavelo
        Ducati
        E-cells
        E-Glide
        E-Go
        E-JOE
        E-Lux
        Eahora
        Ecco
        Ecomotion
        Ecotric
        EG Bike
        Elby
        Electra
        Electric Bike Co
        Electric Bike Tech
        Emazing
        EMOJO
        Energie
        Enzo
        eProdigy
        Espin
        Eunorau
        Evelo
        EVO
        EZ Pedaler
        Fantic
        Faraday
        Felt
        Fifield
        FLASH
        Flx
        Focus
        Freye
        Fuji
        Gazelle
        GenZe
        Gepida
        Ghost
        Giant
        GM
        Go Power
        GoCycle
        Grace
        Green Bike
        GreenBike
        Haibike
        Harley-Davidson
        Head
        Hercules
        Hilltopper
        Hollandia
        HP Velotechnik
        HPC
        ICE
        iGO
        Italjet
        IWEECH
        iZip
        Jeep
        Jetson
        Joulvert
        Juiced
        Junto
        JupiterBike
        Kalkhoff
        Keola
        Kettler
        KHS
        Kona
        KTM
        Lamborghini
        Lapierre
        Leaos
        Lectric
        Leed
        Lexus
        Liberty Trike
        Liesger
        Lithium
        Liv
        Lombardo
        Luna
        M1-Sporttechnik
        M2S
        Magnum
        Marin
        Marss
        Mate
        Maxfoot
        Merida
        Micargo
        Michael Blast
        MOAR
        MOD
        Mondraker
        Motiv
        Moustache
        Nakto
        NCM
        Neomouv
        Ness
        Nicolai
        Norco
        Ohm
        Olic
        Optibike
        Opus
        Orbea
        Organic Transit
        Outrider
        Oyama
        Pedego
        Pegasus
        PESU
        PFIFF
        Phantom
        Piaggio
        Pinarello
        Pininfarina
        Populo
        Porsche
        Priority
        Prodecotech
        Propella
        Puegot
        Pure
        Qualisports
        QuietKat
        Qwic
        Rad
        Raleigh
        Rambo
        Rattan
        Raymon
        Recon
        Reise & Muller
        Revelo
        Revolve
        Rotwild
        Ruff
        Rungu
        Sblocs
        Schwinn
        Scott
        Skoda
        Smartmotion
        Solex
        SONDORS
        Soul Beach
        Spark
        Specialized
        SSR Motorsports
        Stealth Electric
        Steppenwolf
        Stromer
        Superpedestrian
        Surface 604
        Surly
        Swagtron
        Tern
        Tower Electric
        Trek
        Triobike
        Urban Arrow
        Urban Drivestyle
        Van Moof
        Velec
        VELOKS
        Vintage
        Wallerang
        Wheeler
        Winora
        X-treme
        Xtracycle
        Yamaha
        Yub
        ```
        If none of the above brands match, use 'Unknown' as the value.
        """
        "Examples:\n"
        "Original Text: 'Fuji Feather Petina'\n"
        "Cleaned JSON:\n"
        "{\n  \"brand\": \"Fuji\",\n  \"model\": \"Feather Petina\"\n}\n\n"
        "Original Text: 'Ninebot G30 Max'\n"
        "Cleaned JSON:\n"
        "{\n  \"brand\": \"Ninebot\",\n  \"model\": \"G30 Max\"\n}\n\n"
        "Original Text: 'Trek FX 7.4'\n"
        "Cleaned JSON:\n"
        "{\n  \"brand\": \"Trek\",\n  \"model\": \"FX 7.4\"\n}\n\n"
        "Original Text: 'Custom Bike'\n"
        "Cleaned JSON:\n"
        "{\n  \"brand\": \"Unknown\",\n  \"model\": \"Custom Bike\"\n}\n\n"
        "Original Text:\n"
        f"```\n{text}\n```\n"
    )
    
    try:
        # Generate the output using the LLM
        output = llm.generate(inputs=[[{"role": "user", "content": prompt}]])
        # Extract the generated text
        generated_text = output[0][0]  # Access the first element of the outer and inner list
        print(f"Original Brand_Model: {text}")
        print(f"Generated Brand_Model JSON: {generated_text}")
        # Parse the JSON output
        try:
            cleaned_dict = json.loads(generated_text)
        except json.JSONDecodeError:
            print(f"JSON Decode Error for text: {generated_text}")
            return {"brand": "", "model": ""}
        
        # Validate the parsed dictionary
        if all(key in cleaned_dict for key in ['brand', 'model']):
            return cleaned_dict
        else:
            print(f"Missing keys in JSON: {cleaned_dict}")
            return {"brand": "", "model": ""}
    except Exception as e:
        print(f"Error processing Brand_Model: {e}")
        return {"brand": "", "model": ""}  # Fallback to empty if LLM fails

# Initialize the LLM with the appropriate model and schema
llm_note = OpenAILLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    structured_output={"schema": CleanedNote},
    base_url="http://localhost:8000/v1",
)

llm_brand_model = OpenAILLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    structured_output={"schema": CleanedBrandModel},
    base_url="http://localhost:8000/v1",
)

llm_note.load()
llm_brand_model.load()

# Load the data
data_path = Path('your_repair_shop_data.parquet')
df = pd.read_parquet(data_path)

# Filter rows where 'Work Order Internal Note' is not null, not empty, and contains a date
df = df[df['Work Order Internal Note'].notna() & 
        (df['Work Order Internal Note'] != '') & 
        df['Work Order Internal Note'].str.contains(r'\d{1,2}/\d{1,2}/\d{4}')]

# Ensure the required columns exist
required_columns = ['Work Order Internal Note', 'Brand and model ']
for col in required_columns:
    if col not in df.columns:
        # Print all column titles
        print("Columns in the DataFrame:")
        for column in df.columns:
            print(f"- {column}")
        raise ValueError(f"The DataFrame does not contain a '{col}' column.")

# Define prefixes and descriptors for Brand and Model cleaning
brand_prefixes = ['EBK', 'BK', 'ES', 'EB', 'Ebk', 'Es', 'bk', 'es', 'EBK-', 'BK-', 'ES-', 'EB-']
brand_descriptors = [
    'black', 'white', 'grey', 'gray', 'green', 'red', 'blue', 'yellow',
    'orange', 'purple', 'pink', 'silver', 'gold', 'maroon', 'brown',
    'teal', 'cyan', 'cream', 'beige', 'navy', 'bronze', 'lime', 'magenta',
    'turquoise', 'tan', 'dark', 'light', 'with', 'and', 'edition', 'editoin',
    'pro', 'plus', 'v2', 'v3', 'v4', 'v5', 'series', 'model', 'trim',
    'kit', 'upgrade', 'copy', 'folder', 'step', 'through', 'w/', 'without',
    'non-motor', 'multi', 'unknown', 'split', 'fold', 'flip', 'new',
    # Add more descriptors as needed
]

# Function to process a single Brand and Model entry
def process_brand_model(entry: str) -> Optional[dict]:
    if pd.isna(entry):
        return {"brand": "", "model": ""}
    
    # Step 1: Remove prefixes
    text_no_prefix = remove_brand_prefix(entry, brand_prefixes)
    
    # Step 2: Remove descriptors
    text_cleaned = remove_brand_descriptors(text_no_prefix, brand_descriptors)
    
    # Step 3: Use LLM to extract brand and model
    extracted = extract_brand_model_llm(text_cleaned, llm_brand_model)
    
    return extracted

# Function to process a single note
def process_note(note: str) -> Optional[dict]:
    if pd.isna(note):
        return {"cleaned_text": "", "author": "", "date": ""}
    
    # Prepare the prompt for the LLM
    prompt = (
        "Clean the following work order internal note by separating the cleaned text, author, and date. "
        "Return the cleaned text, author, and date in JSON format with fields 'cleaned_text', 'author', and 'date'.\n\n"
        "Examples:\n"
        "Example Original Note:\n"
        "```\n"
        "attempt to fix shifter case\n"
        "Dimitri Politis (6/10/2021 12:03:53)"
        "```\n"
        "Example Cleaned Note:\n"
        "{\n  \"cleaned_text\": \"attempt to fix shifter case\",\n  \"author\": \"Mathieu Gosbee\",\n  \"date\": \"6/10/2021 12:03:53\"\n}\n\n"
        "Example Original Note:\n"
        "```\n"
        "Killian McAteer (16/12/2021 18:23:10)\n"
        "CX bought a brake lever from us and is now loose.\n"
        "Take a deeper look at the front wheel too.\n"
        "```\n"
        "Example Cleaned Note:\n"
        "{\n  \"cleaned_text\": \"CX bought a brake lever from us and is now loose.\\nTake a deeper look at the front wheel too.\",\n  \"author\": \"Mathieu Gosbee\",\n  \"date\": \"16/12/2021 18:23:10\"\n}\n\n"
        "Example Note with missing fields:\n"
        "```\n"
        "CX bought a brake lever from us and is now loose.\n"
        "```\n"
        "Example Cleaned Note with missing fields:\n"
        "{\n  \"cleaned_text\": \"CX bought a brake lever from us and is now loose.\",\n  \"author\": \"\",\n  \"date\": \"\"\n}\n\n"
        f"Original Note:\n```\n{note}\n```\n"
    )
    
    try:
        # Generate the output using the LLM
        output = llm_note.generate(inputs=[[{"role": "user", "content": prompt}], [{"role": "user", "content": prompt}]])
        # Extract the generated text
        generated_text = output[0][0]  # Access the first element of the outer and inner list
        print(f"Original Note: {note}")
        print(f"Generated Note JSON: {generated_text}")
        # Parse the JSON output
        try:
            cleaned_dict = json.loads(generated_text)
        except json.JSONDecodeError:
            print(f"JSON Decode Error for note: {generated_text}")
            return {"cleaned_text": preprocess_text(note), "author": "", "date": ""}
        
        # Validate the parsed dictionary
        if all(key in cleaned_dict for key in ['cleaned_text', 'author', 'date']):
            return cleaned_dict
        else:
            print(f"Missing keys in JSON: {cleaned_dict}")
            return {"cleaned_text": preprocess_text(note), "author": "", "date": ""}
    except Exception as e:
        print(f"Error processing note: {e}")
        return {"cleaned_text": preprocess_text(note), "author": "", "date": ""}  # Fallback to original text if LLM fails

# Apply the processing to the 'Work Order Internal Note' column
# df['Processed Note'] = df['Work Order Internal Note'].apply(lambda note: process_note(note) if note else {"cleaned_text": "", "author": "", "date": ""})

# # Split the 'Processed Note' into separate columns
# df['Cleaned Work Order Internal Note'] = df['Processed Note'].apply(lambda x: x['cleaned_text'] if x else None)
# df['Note Author'] = df['Processed Note'].apply(lambda x: x['author'] if x else None)
# df['Note Date'] = df['Processed Note'].apply(lambda x: x['date'] if x else None)

# # Drop the temporary 'Processed Note' column
# df = df.drop(columns=['Processed Note'])

# Apply the processing to the 'Brand and model ' column
df['Processed Brand_Model'] = df['Brand and model '].apply(lambda entry: process_brand_model(entry) if entry else {"brand": "", "model": ""})

# Split the 'Processed Brand_Model' into separate columns
df['Cleaned Brand'] = df['Processed Brand_Model'].apply(lambda x: x['brand'] if x else None)
df['Cleaned Model'] = df['Processed Brand_Model'].apply(lambda x: x['model'] if x else None)

# Drop the temporary 'Processed Brand_Model' column
df = df.drop(columns=['Processed Brand_Model'])

# Optionally, save the cleaned DataFrame to a new file
cleaned_data_path = Path('your_repair_shop_data_cleaned_more.parquet')
df.to_parquet(cleaned_data_path, index=False)

print("Cleaning complete. Cleaned data saved to 'your_repair_shop_data_cleaned_more.parquet'.")
