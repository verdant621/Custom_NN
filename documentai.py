from google.cloud import documentai_v1beta3 as documentai

def parse_form(project_id='your-project-id',
               input_uri='gs://cloud-samples-data/documentai/form.pdf'):
    """Parse a form"""

    client = documentai.DocumentUnderstandingServiceClient()

    gcs_source = documentai.types.GcsSource(uri=input_uri)

    # mime_type can be application/pdf, image/tiff, and image/gif, or application/json
    input_config = documentai.types.InputConfig(
        gcs_source=gcs_source, mime_type='application/pdf')

    # Setting enabled=True enables form extraction
    form_extraction_params = documentai.types.FormExtractionParams(
        enabled=True)

    # Location can be 'us' or 'eu'
    parent = 'projects/{}/locations/us'.format(project_id)
    request = documentai.types.ProcessDocumentRequest(
        parent=parent,
        input_config=input_config,
        form_extraction_params=form_extraction_params)

    document = client.process_document(request=request)

    def _get_text(el):
        """Doc AI identifies form fields by their offsets 
        in document text. This function converts offsets 
        to text snippets.
        """
        response = ''
        # If a text segment spans several lines, it will
        # be stored in different text segments.
        for segment in el.text_anchor.text_segments:
            start_index = segment.start_index
            end_index = segment.end_index
            response += document.text[start_index:end_index]
        return response

    for page in document.pages:
        print('Page number: {}'.format(page.page_number))
        for form_field in page.form_fields:
            print('Field Name: {}\tConfidence: {}'.format(
                _get_text(form_field.field_name),
                form_field.field_name.confidence))
            print('Field Value: {}\tConfidence: {}'.format(
                _get_text(form_field.field_value),
                form_field.field_value.confidence))

parse_form()
