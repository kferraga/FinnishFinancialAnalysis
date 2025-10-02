import pandas as pd
from bs4 import BeautifulSoup

def create_base_dataframe(csv_loc=None):
    if not csv_loc:
        basic_info_dict = {"name":[], "gender":[], "age":[], "party":[], "occupation":[]}
        financial_info_dict = {"name":[], "1. Total of election campaign expenses":[], "2. Total of election campaign funding": []}
        financial_info = pd.DataFrame()
    return financial_info

def extract_financial_information(html):
    """Given a candidates funding disclosure, scrape relevant information."""
    pass

if __name__ == '__main__':
    print(create_base_dataframe)

# Generate basic dataframe layout OR load it from existing files
# Extract information from municipal election website
