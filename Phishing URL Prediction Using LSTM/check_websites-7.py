#!/usr/bin/env python3
"""This script creates a dataset with the detection of vulnerabilidates in the
websites."""

import csv
import re
import requests
import ssl
import urllib.request
from config import WEBSITES_FILE, OUTPUT_FILE, BROWSER
from localstorage import LocalStorage


def check_sql_inyection(url):
    """Return True if URL is vulnerable to sql injection, False if not."""
    check_url = url + "=1\' or \'1\' = \'1\'"
    try:
        resp = urllib.request.urlopen(check_url)
    except ssl.CertificateError:
        return False
    except urllib.error.HTTPError as e:
        if e.code in (400, 406):
            return False
        else:
            raise
    body      = resp.read()
    text_body = body.decode("utf-8")
    if "you have an error in your sql syntax" in text_body.lower():
        return True
    else:
        return False


def check_certificate_verification(url):
    """Return True if URL is verified, False if not."""
    try:
        requests.get(url, verify = True)
    except requests.exceptions.ConnectionError:
        return False
    else:
        return True


def check_ssl(url, port = 443):
    """Return True if the URL has an SSL certificate, False if not."""
    try:
        cert = ssl.get_server_certificate((url, port))
    except Exception:
        return False
    if not cert:
        return False
    return True


def check_url_syntax(url):
    """Return True if the URL syntax is correct, False if not."""
    regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ... or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$", re.IGNORECASE)

    return re.match(regex, url) is not None


def check_url_online(url):
    """Return True if URL is online, False if not."""
    try:
        urllib.request.urlopen(url)
        return True
    except urllib.request.URLError:
        return False
    except ssl.CertificateError:
        return True


def check_input_validation(url):
    """Check input validation on URL."""
    return not check_url_online(url)


def check_websites(websites_file, output_file):
    """Create a dataset with the vulnerabilities of each URL."""

    print("\n *** CHECKING FOR WEBSITE VULNERABILITIES ***\n")
    with open(output_file, "w") as foutput:
        output = csv.writer(foutput)
        output.writerow(["URL", "HAVE_SSL", "ALLOW_SQL_INY",
                         "HAVE_CERT_VER", "ALLOW_INP_VAL", "UNSAFE"])

        with open(websites_file) as fwebsites:
            websites = csv.reader(fwebsites)
            for i, website in enumerate(websites):
                url = website[0]
                if check_url_syntax(url):
                    if check_url_online(url):
                        print("{} - {}".format(i + 1, url), end = "")
                        try:
                            ssl       = int(check_ssl(url))
                            sql       = int(check_sql_inyection(url))
                            cert     = int(check_certificate_verification(url))
                            input_val = int(check_input_validation(url))
                            unsafe    = int(website[1])
                            output.writerow([url, ssl, sql, cert, input_val,
                                            unsafe])
                        except urllib.error.HTTPError as e:
                            print(" -> HTTP ERROR {}: {}.".format(e.code,
                                                             e.msg.upper()))
                        except urllib.error.URLError as e:
                           print(" -> URL ERROR {}: {}.".format(e.errno,
                                                             e.reason))
                        else:
                            print(" -> OK, SAVED IN DATASET.")
                            # print("CHANGES IN THE LOCAL STORAGE WEB BROWSER:")
                            # local_storage       = LocalStorage(BROWSER, url)
                            # local_storage_items = local_storage.items()
                            # local_storage.quit()
                            # if local_storage_items:
                            #     for key, value in local_storage_items.items():
                            #         print("KEY:", key, "->", value)
                            # else:
                            #     print("No changes.")
                            # print("")
                    else:
                        print("{} - {} -> IT'S NOT ONLINE.".format(i + 1, url))
                else:
                    print("{} - {} -> URL BADLY FORMED.".format(i + 1, url))

    print("\n*** A DATASET HAS BEEN CREATED IN THE {} FILE.\n".format(
          output_file))


if __name__ == "__main__":
    check_websites(WEBSITES_FILE, OUTPUT_FILE)
