import logging
from docx import Document
from docx.shared import Pt
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.oxml.shared import qn
from docx.oxml.ns import nsdecls
from docx.oxml.parser import parse_xml
from typing import Dict, Any

from ..models import DocumentConfig, ProcessorError
from ..utils import parse_size

logger = logging.getLogger(__name__)


class StyleProcessor:
    """Обработчик стилей документа."""

    def __init__(self, doc: Document, config: DocumentConfig):
        """
        Инициализация процессора стилей.

        Args:
            doc: Объект документа.
            config: Конфигурация (типизированный объект).
        """
        self.doc = doc
        self.config = config
        self.logger = logger

    def apply(self) -> None:
        """Применение всех стилей из конфигурации."""
        try:
            self.logger.info("Начало применения стилей")
            self._setup_base_styles()
            self._setup_heading_styles()
            self._setup_special_styles()
            self._apply_line_spacing()
            self._apply_to_existing_document()
            self.logger.info("Стили успешно применены")
        except Exception as e:
            self.logger.error(f"Ошибка применения стилей: {e}")
            raise ProcessorError(f"Ошибка применения стилей: {e}")

    def _setup_base_styles(self) -> None:
        """Настройка основных стилей документа."""
        fonts_cfg = self.config.general.fonts

        # Основной стиль текста
        main_style = self._get_or_create_style(
            style_name='Custom_Main',
            style_type=WD_STYLE_TYPE.PARAGRAPH,
            base_style='Normal'
        )
        self._apply_font_settings(main_style, fonts_cfg['main'])

        # Стиль для приложений
        if 'appendices' in fonts_cfg:
            appendix_style = self._get_or_create_style(
                style_name='Custom_Appendix',
                style_type=WD_STYLE_TYPE.PARAGRAPH,
                base_style='Normal'
            )
            self._apply_font_settings(appendix_style, fonts_cfg['appendices'])

        # Стиль для примечаний
        if 'notes' in fonts_cfg:
            notes_style = self._get_or_create_style(
                style_name='Custom_Notes',
                style_type=WD_STYLE_TYPE.PARAGRAPH,
                base_style='Normal'
            )
            self._apply_font_settings(notes_style, fonts_cfg['notes'])

    def _setup_heading_styles(self) -> None:
        """Настройка стилей заголовков согласно иерархии."""
        fonts_cfg = self.config.general.fonts
        header_num = fonts_cfg.get('headerNum', 3)

        for level in range(header_num):
            style_name = f'Heading {level + 1}'
            heading_style = self._get_or_create_style(
                style_name=style_name,
                style_type=WD_STYLE_TYPE.PARAGRAPH,
                base_style=None
            )

            # Применяем настройки конкретного уровня
            header_font_key = f'header{level + 1}'
            if header_font_key in fonts_cfg:
                font_settings = fonts_cfg[header_font_key].copy()
                self._apply_font_settings(heading_style, font_settings)

            # Настройка отступов для заголовков
            paragraph_format = heading_style.paragraph_format
            paragraph_format.space_before = Pt(12)
            paragraph_format.space_after = Pt(6)
            paragraph_format.keep_with_next = True

    def _setup_special_styles(self) -> None:
        """Настройка специальных стилей."""
        main_font_family = self.config.general.fonts['main'].get('family', 'Arial')

        # Стиль для элементов титульной страницы
        title_style = self._get_or_create_style(
            style_name='Custom_Title',
            style_type=WD_STYLE_TYPE.PARAGRAPH,
            base_style='Normal'
        )
        title_style.font.name = main_font_family
        title_style.font.size = Pt(14)
        title_style.paragraph_format.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

        # Стиль для колонтитулов
        header_style = self._get_or_create_style(
            style_name='Custom_Header',
            style_type=WD_STYLE_TYPE.PARAGRAPH,
            base_style='Normal'
        )
        header_style.font.name = main_font_family
        header_style.font.size = Pt(10)
        header_style.paragraph_format.alignment = WD_PARAGRAPH_ALIGNMENT.RIGHT

    def _apply_line_spacing(self) -> None:
        """Применение межстрочных интервалов."""
        spacing_cfg = self.config.general.spacing
        line_spacing = float(spacing_cfg.line)

        # Применяем ко всем стилям параграфов
        for style in self.doc.styles:
            if style.type == WD_STYLE_TYPE.PARAGRAPH:
                style.paragraph_format.line_spacing = line_spacing

        # Обработка исключений
        if spacing_cfg.exceptions:
            for exception in spacing_cfg.exceptions:
                if 'first_edition' in exception and exception['first_edition'] == 'single':
                    first_edition_style = self._get_or_create_style(
                        style_name='Custom_FirstEdition',
                        style_type=WD_STYLE_TYPE.PARAGRAPH,
                        base_style='Normal'
                    )
                    first_edition_style.paragraph_format.line_spacing = 1.0

    def _apply_to_existing_document(self) -> None:
        """Применяет стили ко всем существующим параграфам документа."""
        main_font_family = self.config.general.fonts['main'].get('family', 'Arial')
        fonts_cfg = self.config.general.fonts

        for paragraph in self.doc.paragraphs:
            # Определяем, какой шрифт применить в зависимости от стиля
            font_family = main_font_family
            font_size = None
            font_bold = None
            
            # Проверяем стиль параграфа
            para_style_name = paragraph.style.name
            if para_style_name.startswith('Heading'):
                # Для заголовков используем специфичные настройки
                try:
                    # Извлекаем уровень из "Heading 1", "Heading 2" и т.д.
                    level = int(para_style_name.split()[-1])
                    header_key = f'header{level}'
                    if header_key in fonts_cfg:
                        font_family = fonts_cfg[header_key].get('family', main_font_family)
                        font_size = parse_size(fonts_cfg[header_key].get('size', '12pt'))
                        font_bold = fonts_cfg[header_key].get('bold', False)
                except (ValueError, IndexError):
                    pass
            
            # Применяем шрифт ко всем runs в параграфе
            for run in paragraph.runs:
                run.font.name = font_family
                if font_size:
                    run.font.size = font_size
                if font_bold is not None:
                    run.font.bold = font_bold

    def _get_or_create_style(self, style_name: str, style_type: int, base_style: str = None):
        """Получает или создает стиль с указанными параметрами."""
        try:
            style = self.doc.styles[style_name]
        except KeyError:
            style = self.doc.styles.add_style(style_name, style_type)
            if base_style:
                style.base_style = self.doc.styles[base_style]
        return style

    def _apply_font_settings(self, style, font_cfg: Dict[str, Any]) -> None:
        """Применяет настройки шрифта к стилю."""
        if 'family' in font_cfg:
            self._set_font_family(style, font_cfg['family'])

        if 'size' in font_cfg:
            size_pt = parse_size(font_cfg['size'])
            self._set_font_size(style, size_pt)

        if 'bold' in font_cfg:
            self._set_font_bold(style, font_cfg['bold'])

        if 'italic' in font_cfg:
            self._set_font_italic(style, font_cfg['italic'])

    def _set_font_family(self, style, family: str) -> None:
        """Устанавливает семейство шрифта."""
        if style.type == WD_STYLE_TYPE.PARAGRAPH:
            pPr = style.element.get_or_add_pPr()
            rPr = pPr.find(qn('w:rPr'))
            if rPr is None:
                rPr_xml = f'<w:rPr {nsdecls("w")}></w:rPr>'
                rPr = parse_xml(rPr_xml)
                pPr.append(rPr)

            rFonts = rPr.find(qn('w:rFonts'))
            if rFonts is None:
                rFonts_xml = f'<w:rFonts {nsdecls("w")}></w:rFonts>'
                rFonts = parse_xml(rFonts_xml)
                rPr.append(rFonts)

            rFonts.set(qn('w:ascii'), family)
            rFonts.set(qn('w:hAnsi'), family)
            rFonts.set(qn('w:cs'), family)
        else:
            style.font.name = family

    def _set_font_size(self, style, size_pt: float) -> None:
        """Устанавливает размер шрифта."""
        if style.type == WD_STYLE_TYPE.PARAGRAPH:
            pPr = style.element.get_or_add_pPr()
            rPr = pPr.find(qn('w:rPr'))
            if rPr is None:
                rPr_xml = f'<w:rPr {nsdecls("w")}></w:rPr>'
                rPr = parse_xml(rPr_xml)
                pPr.append(rPr)

            # Размер шрифта
            sz_elem = rPr.find(qn('w:sz'))
            if sz_elem is None:
                sz_xml = f'<w:sz {nsdecls("w")} w:val="{int(size_pt * 2)}"/>'
                sz_elem = parse_xml(sz_xml)
                rPr.append(sz_elem)
            else:
                sz_elem.set(qn('w:val'), str(int(size_pt * 2)))

            # Размер для комплексных скриптов
            szCs_elem = rPr.find(qn('w:szCs'))
            if szCs_elem is None:
                szCs_xml = f'<w:szCs {nsdecls("w")} w:val="{int(size_pt * 2)}"/>'
                szCs_elem = parse_xml(szCs_xml)
                rPr.append(szCs_elem)
            else:
                szCs_elem.set(qn('w:val'), str(int(size_pt * 2)))
        else:
            style.font.size = Pt(size_pt)

    def _set_font_bold(self, style, is_bold: bool) -> None:
        """Устанавливает жирность шрифта."""
        if style.type == WD_STYLE_TYPE.PARAGRAPH:
            pPr = style.element.get_or_add_pPr()
            rPr = pPr.find(qn('w:rPr'))
            if rPr is None:
                rPr_xml = f'<w:rPr {nsdecls("w")}></w:rPr>'
                rPr = parse_xml(rPr_xml)
                pPr.append(rPr)

            if is_bold:
                b_elem = rPr.find(qn('w:b'))
                if b_elem is None:
                    b_xml = f'<w:b {nsdecls("w")}/>'
                    b_elem = parse_xml(b_xml)
                    rPr.append(b_elem)

                bCs_elem = rPr.find(qn('w:bCs'))
                if bCs_elem is None:
                    bCs_xml = f'<w:bCs {nsdecls("w")}/>'
                    bCs_elem = parse_xml(bCs_xml)
                    rPr.append(bCs_elem)
            else:
                for elem in rPr.findall(qn('w:b')):
                    rPr.remove(elem)
                for elem in rPr.findall(qn('w:bCs')):
                    rPr.remove(elem)
        else:
            style.font.bold = is_bold

    def _set_font_italic(self, style, is_italic: bool) -> None:
        """Устанавливает курсив шрифта."""
        if style.type == WD_STYLE_TYPE.PARAGRAPH:
            pPr = style.element.get_or_add_pPr()
            rPr = pPr.find(qn('w:rPr'))
            if rPr is None:
                rPr_xml = f'<w:rPr {nsdecls("w")}></w:rPr>'
                rPr = parse_xml(rPr_xml)
                pPr.append(rPr)

            if is_italic:
                i_elem = rPr.find(qn('w:i'))
                if i_elem is None:
                    i_xml = f'<w:i {nsdecls("w")}/>'
                    i_elem = parse_xml(i_xml)
                    rPr.append(i_elem)

                iCs_elem = rPr.find(qn('w:iCs'))
                if iCs_elem is None:
                    iCs_xml = f'<w:iCs {nsdecls("w")}/>'
                    iCs_elem = parse_xml(iCs_xml)
                    rPr.append(iCs_elem)
            else:
                for elem in rPr.findall(qn('w:i')):
                    rPr.remove(elem)
                for elem in rPr.findall(qn('w:iCs')):
                    rPr.remove(elem)
        else:
            style.font.italic = is_italic
