from doc_editor.editor import DocumentEditor


# Компактный демонстрационный файл для научной статьи
# Использует гибридный конфиг со всеми функциями
editor = DocumentEditor("doc_editor/tests/test_data/sb_14.docx")
editor.load_config("doc_editor/tests/test_data/formatConfig_with_formatted_headers.yaml")
editor.apply_config()
editor.save("output_demo_compact_14.docx")


