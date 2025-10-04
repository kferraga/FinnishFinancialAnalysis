import pandas as pd
import yaml
from bs4 import BeautifulSoup
import requests

def create_base_dataframe(csv_locs=None):
    if not csv_loc:
        basic_info_dict = {"name":[], "gender":[], "age":[], "party":[], "occupation":[]}
        financial_info_dict = {"name":[], "1. Total of election campaign expenses":[], "2. Total of election campaign funding": []}
        financial_info = pd.DataFrame()
    else:
        basic_info_dict = pd.read_csv(csv_loc)
    return financial_info

def extract_financial_information(election="", regions=[]):
    """Given candidate funding disclosures from vaalirahoitusvalvonta.fi, scrape relevant information.
    :param election: An election to search, in the format "kuntavaalit2021".
    The available options are:
    - aluevaalitXXXX (county elections);
    - kuntavaalitXXXX (municipal elections);
    - eduskuntavaalitXXXX (parliamentary elections);
    - europarlamenttivaalitXXXX (European elections);
    - presidentinvaaliXXXX (presidential elections).
    The website only contains reports online for ~5-7 years, dependent on the election type.
    :param regions: A list of regions to search. Can also set it to "all" to go through all available.
    """
    base_url = "https://www.vaalirahoitusvalvonta.fi/en/frontpage/electioncampaignfunding/disclosuresbyelection/"
    try:
        page = requests.get(base_url+election+".html")
        soup = BeautifulSoup(page.content, "html.parser")
        results = soup.find(class_="julkinenpalvelu-article-header")
        print(results.prettify())
    except Exception as e:
        print(e)
    return page

def load_yaml_file(file_path):
    """Quick function to load and return a yaml file given a file path."""
    access = open(file_path, "r")
    yaml_file = yaml.safe_load(access)
    access.close()
    return yaml_file

if __name__ == '__main__':
    # We have a list of CSV files and a list of potential variables from the YAML file. Grab all of the variables from the
    # CSV files and put them into a set. Error check to see if all values exist in the YAML file
    # Then, when we extract from the HTML based on the YAML file, apply into the CSV file based on ID

    yaml_file = load_yaml_file("fin_elec_municipality_candidate_info_and_votes.yml")
    extract_financial_information("kuntavaalit2025")
    print("to")

    # csv_locs = []
    # print(create_base_dataframe)

# Generate basic dataframe layout OR load it from existing files
# Extract information from municipal election website
