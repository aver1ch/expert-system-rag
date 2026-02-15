from flask import Flask, request, jsonify, send_file
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
import yaml
import requests
import os
import uuid
import tempfile
from urllib.parse import urlparse
from doc_editor.editor import DocumentEditor

import logging
# pipreqs . --force
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path


def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


@app.route('/api/process_document', methods=['POST'])
def process_document():
    # Получаем данные из запроса
    data = request.json
    if not data or 'document_url' not in data or 'config' not in data:
        return jsonify({'error': 'Missing document_url or config in request'}), 400

    document_url = data['document_url']
    config_data = data['config']

    # Валидация URL
    #if not is_valid_url(document_url):
     #   return jsonify({'error': 'Invalid document URL'}), 400

    # Парсинг конфига
    try:
        config = yaml.safe_load(config_data)
    except yaml.YAMLError as e:
        return jsonify({'error': f'Invalid YAML config: {str(e)}'}), 400

    # Создаем временные файлы
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Скачиваем документ
            input_path = os.path.join(temp_dir, 'input.docx')
            config_path = os.path.join(temp_dir, 'config.yaml')
            download_file(document_url, input_path)
            download_file(config_data, config_path)

            # Обрабатываем документ
            output_path = os.path.join(temp_dir, 'output.docx')
            editor = DocumentEditor(input_path)
            editor.load_config(config_path)
            editor.apply_config()
            editor.save(output_path)

            # Отправляем результат
            return send_file(
                output_path,
                as_attachment=True,
                download_name='formatted_document.docx',
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            )
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'Failed to download document: {str(e)}'}), 400
        except Exception as e:
            return jsonify({'error': f'Document processing failed: {str(e)}'}), 500
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)