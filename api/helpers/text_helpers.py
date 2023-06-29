import csv
import io

import pypdf
from langchain.schema import Document
from typing import List


def get_text_from_pdf(contents) -> str:
    pdf_reader = pypdf.PdfReader(io.BytesIO(contents))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def get_text_from_txt(contents) -> str:
    return contents.decode()


# TODO make this have more validation on format of CSV such that
# we know metadata is correct.
def get_text_from_csv(contents) -> List[Document]:
    decoded_contents = contents.decode('utf-8')
    docs = []

    # Create a CSV reader object
    reader = csv.reader(decoded_contents.splitlines())

    # Skip the header
    next(reader)

    for row in reader:
        doc = Document(
            page_content=row[1],
            metadata={
                "title": row[0],
                "url": row[2],
            })
        docs.append(doc)

    return docs

def parse_files(files, logger):
    """Parse the files and extract the text into a `texts[]` or `docs[]`"""
    texts: List[str] = []
    fnames: List[str] = []
    docs: List[Document] = []

    for file in files:
        logger.info("Reading file: {}".format(file.filename))
        contents = file.file.read()
        # PDF file
        if file.content_type == 'application/pdf':
            logger.info("File {} is a PDF".format(file.filename))
            pdfText = get_text_from_pdf(contents)

            texts.append(pdfText)
            fnames.append(file.filename)
        # Text file
        elif file.content_type == 'text/plain':
            logger.info("File {} is a TXT".format(file.filename))
            txtText = get_text_from_txt(contents)

            texts.append(txtText)
            fnames.append(file.filename)
        # CSV files
        # TODO make this more robust with document thing — currently it's just text on the whole thing
        elif file.content_type == 'text/csv':
            logger.info("File {} is a CSV".format(file.filename))
            csvDocs = get_text_from_csv(contents)

            docs.extend(csvDocs)
            fnames.append(file.filename)
        else:
            logger.warning(
                "Unsupported file type for file: {}".format(file.filename))

    return texts, docs
