"""Transfer explicitly selected project files; never print signed credentials."""
import concurrent.futures
import hashlib
import http.client
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen
import uuid

CONFIG = Path("/private/tmp/gci_notion_transfer_20261025.json")
ROOT = Path(__file__).resolve().parent.parent


def download(item):
    version = "v16" if "v16" in item["name"] else "v17"
    target = ROOT / "entregaveis" / version / item["name"]
    with urlopen(item["url"], timeout=90) as response:
        data = response.read(25 * 1024 * 1024 + 1)
    assert len(data) <= 25 * 1024 * 1024, "Arquivo maior que o limite desta recuperação."
    if target.exists():
        assert target.read_bytes() == data, "Arquivo local diferente: preservar antes de substituir."
    else:
        target.write_bytes(data)
    return {"name": item["name"], "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def upload(config, path, output_name="notion_upload_v18_zip.json"):
    boundary = "gci-" + uuid.uuid4().hex
    data = path.read_bytes()
    prefix = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{config["filename"]}"\r\n'
              f'Content-Type: {config["content_type"]}\r\n\r\n').encode()
    body = prefix + data + f"\r\n--{boundary}--\r\n".encode()
    address = urlsplit(config["upload_url"])
    connection = http.client.HTTPSConnection(address.hostname, timeout=90)
    headers = dict(config["upload_headers"])
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    headers["Content-Length"] = str(len(body))
    # Exactly one POST for the one-time upload URL; no automatic retries.
    connection.request("POST", address.path + ("?" + address.query if address.query else ""), body, headers)
    response = connection.getresponse()
    response_body = response.read()
    connection.close()
    result = json.loads(response_body)
    result["http_status"] = response.status
    output = ROOT / "entregaveis" / output_name
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    assert 200 <= response.status < 300, f"Upload retornou HTTP {response.status}; resposta salva, não repetir sem verificar."
    print("UPLOAD", path.name, "HTTP", response.status)


if __name__ == "__main__":
    config = json.loads(CONFIG.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(download, config["downloads"]):
            print(json.dumps(result))
    upload(config["upload"], ROOT / "entregaveis/v18/gci_home_credit_v18.zip")
