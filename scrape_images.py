import os
root = os.getcwd()

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

import time

driver_path = '/Users/davidchung/desktop/chromedriver'
import urllib.request

#  Make root directory for scraping image data
imgs_path = os.path.join(root, 'poke_imgs')
if not os.path.isdir(imgs_path):
    os.mkdir(imgs_path)

#  Import pokemon dataframe, get list of pokemon to import (Kanto only)
# pokemon = pd.read_csv('pokemon.csv')
# poke_list = pokemon.Name.values[:151]
poke_list = ['bulbasaur', 'charmander', 'squirtle', 'ivysaur', 'charmeleon', 'wartortle', 'venusaur', 'charizard', 'blastoise']
for query in poke_list:  #  Make directory for each query
    if query in ['bulbasaur', 'charmander']:
        continue
    poke_path = os.path.join(imgs_path, query)
    if not os.path.isdir(poke_path):
            os.mkdir(poke_path)
    wd = webdriver.Chrome(driver_path)  #  Find images for the query
    wd.get('https://images.google.com')
    searchbar = wd.find_element_by_name('q')
    searchbar.send_keys(query + ' pokemon images' + '\n')

    for i in range(20): #  Scroll to bottom of page to display more images on web page
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        try:
            expand_results = wd.find_element_by_class_name("mye4qd")
            expand_results.click()
        except:
            print('Continuing to scroll down!')
        time.sleep(3)

    imgs = wd.find_elements_by_xpath('//img[@class="rg_i Q4LuWd"]')
    print(len(imgs), 'images found.')
    print("Getting images for " + query.capitalize() + '...')
    count = 0
    for idx, j in enumerate(imgs):
        link = j.get_attribute('src')
        try:
            response = urllib.request.urlretrieve(link, os.path.join(poke_path, f"{idx+1}" + '.jpg'))
            count += 1
            print('Saved image!')
        except:
            print('Unable to retrieve image...')
    print('Found ' + f"{count}" + ' images for ' + query.capitalize() + '.')
    wd.close()
