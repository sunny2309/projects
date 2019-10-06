from selenium import webdriver
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

DRIVER = '/home/sunny/Desktop/chromedriver' ## Path to chrome driver. it should match same version as that of chrome browser.
## Download chrome driver from this location : https://chromedriver.chromium.org/downloads

options = webdriver.ChromeOptions() # define options
options.add_argument("headless") 

driver = webdriver.Chrome(DRIVER, options=options)
driver.set_window_size(2400, 2000)

if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

with open(sys.argv[1]) as f:
    for line in f:
        image_name  = 'screenshots/'+ line.split(',')[0].strip()+'.png'
        #print(image_name)
        url_name = line.split(',')[1].strip().lstrip('/')
        if not (url_name.startswith('http') or url_name.startswith('https')):
            url_name = 'http://' + url_name
        #print(url_name.strip())
        driver.get(url_name.strip())
        if 'vimeo' in url_name:
            time.sleep(10)
        elif 'youtube' in url_name:
            time.sleep(5)
        else:
            time.sleep(3)
        screenshot = driver.save_screenshot(image_name)
driver.quit()
