"""
Created At: 04/07/2021 22:35
"""
import pandas as pd
import requests

headers = {'User-Agent': 'Mozilla/5.0'}
data = {'keyword':'',
        'pageNo': 1,
        'itemsPerPage': 10,
        'layout': 'Decl.DataSet.Detail.default',
        'service': 'Content.Decl.DataSet.Grouping.select',
        'itemId': '60d29c8730817e57a33c3173',
        'gridModuleParentId': 12,
        'type': 'Decl.DataSet',
        'page': '',
        'modulePosition': 0,
        'moduleParentId': -1,
        'orderBy': '',
        'unRegex': '',
        '_t': '1625409875815'}

result = None
num_of_requests = 10
dead_server_txt = '503 Service Temporarily Unavailable'
not_found_list = []
URL = "https://namdinh.edu.vn/?module=Content.Listing&moduleId=1012&cmd=redraw&site=19012&url_mode=rewrite&submitFormId=1012&moduleId=1012&page=&site=19012"


for i in range(1, 498):
  id = str(i)
  student_id = '600' + '0'*(3 - len(id)) + id
  data['keyword'] = student_id
  r = requests.post(url = URL,headers=headers, data = data)
  print(student_id)
  if dead_server_txt in r.text:
    is_found = False
    for j in range(num_of_requests):
      print(j)
      r = requests.post(url = URL,headers=headers, data = data)
      if dead_server_txt not in r.text:
        is_found = True
        break
    if is_found is True:
      df_list = pd.read_html(r.text)
      df = df_list[0]
      if result is None:
        result = df
      else:
        result = pd.concat([result, df], ignore_index=True)
    else:
      not_found_list.append(student_id)
      print('fail')
  else:
    df_list = pd.read_html(r.text)
    df = df_list[0]
    if result is None:
      result = df
    else:
      result = pd.concat([result, df], ignore_index=True)


result.to_csv('file_name.csv', encoding='utf-8')
