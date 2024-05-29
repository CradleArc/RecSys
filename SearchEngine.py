import requests
from bs4 import BeautifulSoup

def aminer_search(query):
    base_url = "https://www.aminer.cn/search/person"
    params = {"q": query, "t": "b"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }

    response = requests.get(base_url, params=params)
    print(response.text)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all("div", class_="search-result-item")
        # print(soup)
        for result in results:
            name = result.find("p", class_="name").text.strip()
            title = result.find("p", class_="position").text.strip()
            affiliation = result.find("p", class_="affiliation").text.strip()
            print("Name:", name)
            print("Title:", title)
            print("Affiliation:", affiliation)
            print("="*50)
    else:
        print("Failed to retrieve search results.")

if __name__ == "__main__":
    query = input("Enter your query: ")
    aminer_search(query)
