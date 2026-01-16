import requests, json, time

payload = {
  "jsonrpc": "2.0",
  "method": "tasks/send",
  "params": {
    "message": {"parts": [{"text": "Please turn on the living room light"}]}
  },
  "id": "1"
}

if __name__ == "__main__":
    r = requests.post('http://127.0.0.1:9010/a2a', json=payload, timeout=5)
    print('send status', r.status_code)
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))

    res = r.json().get('result')
    task_id = res.get('id')
    print('task_id', task_id)

    # Poll
    for i in range(10):
        time.sleep(0.3)
        pr = requests.post('http://127.0.0.1:9010/a2a', json={
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"id": task_id},
            "id": str(i+2)
        }, timeout=5)
        print('poll', i, pr.status_code)
        print(json.dumps(pr.json(), indent=2, ensure_ascii=False))
        if pr.json().get('result', {}).get('status', {}).get('state') == 'completed':
            break
    print('done')
