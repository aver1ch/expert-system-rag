from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class HeaderTextPart:
    """Часть текста колонтитула с форматированием."""
    text: str
    bold: bool = False
    italic: bool = False
    font_family: Optional[str] = None


@dataclass
class FontConfig:
    """Конфигурация шрифта."""
    family: str
    size: str = "12pt"
    bold: bool = False
    italic: bool = False


@dataclass
class MarginsConfig:
    """Конфигурация полей документа."""
    left: str = "20mm"
    right: str = "10mm"
    top: str = "20mm"
    bottom: str = "20mm"


@dataclass
class SpacingConfig:
    """Конфигурация межстрочных интервалов."""
    line: float = 1.5
    exceptions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GeneralConfig:
    """Основные настройки документа."""
    margins: MarginsConfig
    fonts: Dict[str, Any]  # main, appendices, notes, headerNum, header1, header2, etc
    spacing: SpacingConfig


@dataclass
class TableFormatConfig:
    """Конфигурация форматирования таблиц в титуле."""
    preserve_existing: bool = True
    apply_font: bool = True
    apply_spacing: bool = True


@dataclass
class TitlePageConfig:
    """Конфигурация титульного листа."""
    template: str = ""
    template_path: str = ""
    image_path: str = ""
    enabled: bool = True
    elements: List[Dict[str, str]] = field(default_factory=list)
    appendix: str = "А"
    line_spacing: float = 1.5
    spacing_before: float = 0.0
    spacing_after: float = 0.0
    table_format: TableFormatConfig = field(default_factory=TableFormatConfig)


@dataclass
class HeadersConfig:
    """Конфигурация колонтитулов."""
    left: str = ""
    right: str = ""
    page_numbers: bool = False
    enabled: bool = False
    right_parts: List[HeaderTextPart] = field(default_factory=list)
    left_parts: List[HeaderTextPart] = field(default_factory=list)


@dataclass
class SectionConfig:
    """Конфигурация нумерации разделов (Фаза 2)."""
    enabled: bool = True
    start_number: int = 1
    numbering_format: str = "decimal"  # decimal, roman, arabic
    include_in_toc: bool = True
    auto_number_headings: bool = True
    numbering_levels: int = 3  # поддержка уровней нумерации


@dataclass
class TOCConfig:
    """Конфигурация оглавления (Фаза 2)."""
    enabled: bool = False
    title: str = "ОГЛАВЛЕНИЕ"
    page_numbers: bool = True
    levels: int = 3


@dataclass
class PrefaceConfig:
    """Конфигурация предисловия (Фаза 2)."""
    enabled: bool = False
    content: str = ""


@dataclass
class AppendixConfig:
    """Конфигурация приложений (Фаза 2)."""
    enabled: bool = False
    numbering_style: str = "letters"  # letters (A, Б, В...) или numbers (1, 2, 3...)


@dataclass
class NumberingConfig:
    """Конфигурация нумерации."""
    headers: HeadersConfig = field(default_factory=HeadersConfig)
    pages: Optional[Dict[str, Any]] = None
    sections: SectionConfig = field(default_factory=SectionConfig)


@dataclass
class DocumentStructureConfig:
    """Конфигурация структуры документа (Фаза 2)."""
    sections: SectionConfig = field(default_factory=SectionConfig)
    toc: TOCConfig = field(default_factory=TOCConfig)
    preface: PrefaceConfig = field(default_factory=PrefaceConfig)
    appendix: AppendixConfig = field(default_factory=AppendixConfig)


@dataclass
class StructureConfig:
    """Конфигурация структуры документа."""
    title_page: TitlePageConfig = field(default_factory=TitlePageConfig)
    sections: Optional[Dict[str, Any]] = None
    numbering: NumberingConfig = field(default_factory=NumberingConfig)
    document_structure: DocumentStructureConfig = field(default_factory=DocumentStructureConfig)


@dataclass
class DocumentConfig:
    """Полная конфигурация документа."""
    general: GeneralConfig
    structure: StructureConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentConfig":
        """Создаёт объект конфига из словаря."""
        doc_data = data.get('document', {})
        
        # Парсим general
        general_data = doc_data.get('general', {})
        margins_data = general_data.get('margins', {})
        margins = MarginsConfig(**margins_data) if margins_data else MarginsConfig()
        
        fonts_data = general_data.get('fonts', {})
        spacing_data = general_data.get('spacing', {})
        spacing = SpacingConfig(**spacing_data) if spacing_data else SpacingConfig()
        
        general = GeneralConfig(
            margins=margins,
            fonts=fonts_data,
            spacing=spacing
        )
        
        # Парсим structure
        structure_data = doc_data.get('structure', {})
        title_page_data = structure_data.get('title_page', {})
        title_page = TitlePageConfig(**title_page_data) if title_page_data else TitlePageConfig()

        # Нумерация и заголовки находятся на уровне `document.structure.numbering` в YAML
        numbering_data = structure_data.get('numbering', {})
        headers_data = numbering_data.get('headers', {})
        
        # Преобразуем right_parts и left_parts в HeaderTextPart если они присутствуют
        if 'right_parts' in headers_data and isinstance(headers_data['right_parts'], list):
            headers_data['right_parts'] = [
                HeaderTextPart(**part) for part in headers_data['right_parts']
            ]
        if 'left_parts' in headers_data and isinstance(headers_data['left_parts'], list):
            headers_data['left_parts'] = [
                HeaderTextPart(**part) for part in headers_data['left_parts']
            ]
        
        headers = HeadersConfig(**headers_data) if headers_data else HeadersConfig()
        numbering = NumberingConfig(
            headers=headers,
            pages=numbering_data.get('pages')
        )
        
        structure = StructureConfig(
            title_page=title_page,
            sections=structure_data.get('sections'),
            numbering=numbering
        )
        
        return cls(general=general, structure=structure)
