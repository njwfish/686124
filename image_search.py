#!/usr/local/bin/python3.6

from urllib.request import Request, urlopen
from urllib import parse
import shutil
import time
from selenium import webdriver
from selenium.common.exceptions import ElementNotInteractableException


class SeleniumImageSearch:
    """Selenium Image Search: uses Firefox, geckodriver and Selenium to pull Google Images search results"""
    def __init__(self):
        self.i_img = 0

    def download_link(self, url, download_dir):
        """
        Download url to given folder
        :param url: url to download
        :param download_dir: folder to download to
        """
        # Set header to prevent web scraping blocking
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) '
                                                   'AppleWebKit/537.75.14 (KHTML, like Gecko) '
                                                   'Version/7.0.3 Safari/7046A194A'})
        # Attempt to downlad url, if fails, move on
        try:
            with urlopen(req) as response, open(download_dir + str(self.i_img) + '.png', 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            self.i_img += 1
        except:
            pass

    def search(self, query, download_dir, count):
        """
        Uses Firefox to get image links, and download them
        :param query: search to query using google images
        :param download_dir: path to directory to download search results
        :param count: number of results to download
        """
        self.i_img = 0
        url = "https://www.google.com/search?q=" + query + "&source=lnms&tbm=isch"
        # Open Firefox window
        driver = webdriver.Firefox()
        driver.get(url)
        while self.i_img < count:
            # Scroll down and wait to load images
            for scroll in range(10):
                driver.execute_script("window.scrollBy(0,1000000)")
                time.sleep(0.2)
            time.sleep(0.5)
            # Get Google links from page
            images = driver.find_elements_by_xpath("//a[@class='rg_l']")
            # Parse image link from Google result and download
            for image in images:
                if self.i_img >= count:
                    break
                google_url = image.get_attribute("href")
                google_url, complex_url = google_url.split("imgurl=")
                org_url, other_url = complex_url.split("&imgrefurl=")
                org_url = parse.unquote(org_url)
                self.download_link(org_url, download_dir)
            button_smb = driver.find_element_by_xpath("//input[@id='smb']")
            if button_smb is not None:
                try:
                    button_smb.click()
                except ElementNotInteractableException:
                    pass
        # Close Firefox window
        driver.quit()
