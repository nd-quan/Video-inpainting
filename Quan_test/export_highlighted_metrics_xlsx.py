#!/usr/bin/env python3
"""Export comparison CSV as a styled XLSX using only the standard library."""
import csv
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    source = Path(sys.argv[1]).resolve()
    with source.open(encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        rows = list(reader)
    metrics = headers[3:]
    best = {}
    for sequence in dict.fromkeys(row['Sequence'] for row in rows):
        group = [row for row in rows if row['Sequence'] == sequence]
        for metric in metrics:
            best[sequence, metric] = (max if metric in ('PSNR', 'SSIM', 'FS') else min)(
                float(row[metric]) for row in group)
    sheet_rows = []
    highlighted = 0
    for index, row in enumerate([dict(zip(headers, headers)), *rows], start=1):
        cells = []
        for column, header in enumerate(headers):
            ref = f'{chr(65 + column)}{index}'
            value = row[header]
            if index == 1 or column < 2:
                if index > 1 and header == 'Method' and value == 'R×4':
                    value = 'TCE'
                style = 1 if index == 1 else 0
                cells.append(f'<c r="{ref}" s="{style}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
            else:
                number = float(value)
                winner = column >= 3 and math.isclose(number, best[row['Sequence'], header], rel_tol=1e-12, abs_tol=1e-12)
                highlighted += int(winner)
                style = 0 if column == 2 else (2 if header in ('PSNR', 'VFID') else 3)
                if winner:
                    style += 2
                cells.append(f'<c r="{ref}" s="{style}"><v>{number!r}</v></c>')
        sheet_rows.append(f'<row r="{index}" ht="22" customHeight="1">{"".join(cells)}</row>')
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    worksheet = f'''<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="1" width="22" customWidth="1"/><col min="2" max="3" width="12" customWidth="1"/><col min="4" max="10" width="19" customWidth="1"/></cols><sheetData>{''.join(sheet_rows)}</sheetData><autoFilter ref="A1:J{len(rows)+1}"/></worksheet>'''
    styles = f'''<?xml version="1.0" encoding="UTF-8"?>
<styleSheet xmlns="{ns}"><numFmts count="2"><numFmt numFmtId="164" formatCode="0.0000"/><numFmt numFmtId="165" formatCode="0.00000"/></numFmts><fonts count="3"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FF146B35"/><name val="Calibri"/></font></fonts><fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF263746"/><bgColor indexed="64"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFDDF2E3"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="6"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/><xf numFmtId="164" fontId="2" fillId="3" borderId="0" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1"/><xf numFmtId="165" fontId="2" fillId="3" borderId="0" xfId="0" applyNumberFormat="1" applyFont="1" applyFill="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    output = source.with_suffix('.xlsx')
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>')
        archive.writestr('_rels/.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Comparison" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        archive.writestr('xl/styles.xml', styles)
        archive.writestr('xl/worksheets/sheet1.xml', worksheet)
    from xml.etree import ElementTree
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            ElementTree.fromstring(archive.read(name))
    print(f'Saved {output}; {len(rows)} data rows, {highlighted} highlighted cells.')


if __name__ == '__main__':
    main()
