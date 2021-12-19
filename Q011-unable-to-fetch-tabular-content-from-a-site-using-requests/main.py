from pprint import pprint
import webbrowser

import requests

link = 'https://cri.nbb.be/bc9/web/catalog'

payload = {
    'javax.faces.partial.ajax': 'true',
    'javax.faces.source': 'page_searchForm:actions:0:button',
    'javax.faces.partial.execute': 'page_searchForm',
    'javax.faces.partial.render': 'page_searchForm page_listForm pageMessagesId',
    'page_searchForm:actions:0:button': 'page_searchForm:actions:0:button',
    'page_searchForm': 'page_searchForm',
    'page_searchForm:j_id3:generated_number_2_component': '0466425389',
    'page_searchForm:j_id3:generated_name_4_component': '',
    'page_searchForm:j_id3:generated_address_zipCode_6_component': '',
    'page_searchForm:j_id3_activeIndex': '0',
    'page_searchForm:j_id2_stateholder': 'panel_param_visible;',
    'page_searchForm:j_idt133_stateholder': 'panel_param_visible;',
    'javax.faces.ViewState': 'e1s1'
}

headers = {
    'Faces-Request': 'partial/ajax',
    'X-Requested-With': 'XMLHttpRequest',
    'Origin': 'https://cri.nbb.be',
    'Accept': 'application/xml, text/xml, */*; q=0.01',
    'Accept-Encoding': 'gzip, deflate, br',
    'Host': 'cri.nbb.be',
    'Origin': 'https://cri.nbb.be',
    'Referer': 'https://cri.nbb.be/bc9/web/catalog?execution=e1s1'
}


final_get_headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "sec-ch-ua": "\"Google Chrome\";v=\"95\", \"Chromium\";v=\"95\", \";Not A Brand\";v=\"99\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "Referer": "https://cri.nbb.be/bc9/web/catalog?execution=e1s1",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}

with requests.Session() as s:
    s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36'

    res_1 = s.get(link, params={'execution': 'e1s1'})
    pprint(res_1.url)
    print(res_1.status_code)
    pprint(dict(res_1.headers))

    s.headers.update(headers)

    res_2 = s.post(link,
                   data=payload,
                   params={'execution': 'e1s1'})
    pprint(res_2.url)
    print(res_2.status_code)
    pprint(dict(res_2.headers))

    res_3 = s.get(link,
                  params={'execution': 'e1s2'},
                  headers=final_get_headers)

    open('response.html', 'wt').write(res_3.text)
    webbrowser.open('response.html')

    pprint(res_3.url)
    print(res_3.status_code)
    pprint(dict(res_3.headers))
