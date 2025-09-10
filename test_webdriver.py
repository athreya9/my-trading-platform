from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

print("Initializing webdriver...")
try:
    service = Service(
        ChromeDriverManager().install(),
        service_args=["--verbose", "--log-path=chromedriver.log"]
    )
    driver = webdriver.Chrome(service=service)
    print("Webdriver initialized successfully.")
    driver.quit()
    print("Webdriver quit successfully.")
except Exception as e:
    print(f"An error occurred: {e}")