# Main Service Unit Tests - Lead Extraction with AI Agent

This document explains how to use the updated unit tests for the Main Service, which now includes AI-powered lead extraction from images.

## Overview

The Main Service now features an AI agent that can analyze uploaded images and automatically extract lead information. The agent uses OpenAI's GPT-4 Vision model to:

1. Analyze uploaded images for lead-related content (business cards, LinkedIn profiles, contact info, etc.)
2. Extract relevant information (names, emails, social profiles, etc.)
3. Automatically add leads to the CRM system via the MongoDB service

## Prerequisites

Before running the unit tests, ensure you have:

1. **Services Running:**
   - Main Service on port 8000 (default) or 11000 (as configured in tests)
   - MongoDB Service on port 10000
   - MongoDB database running on localhost:27017

2. **Environment Variables:**
   - `OPENAI_API_KEY` - Your OpenAI API key for the AI agent
   - `MONGODB_SERVICE` - MongoDB service URL (optional, defaults to http://127.0.0.1:10000)

3. **Test Images:**
   - Run `python create_test_images.py` to generate sample test images
   - Or prepare your own images containing lead information

## Running the Tests

### Using Postman

1. **Import the Collection:**
   ```
   Import: service_MainService/Unit Tests/MainService_Unit_Tests.postman_collection.json
   ```

2. **Set Environment Variables:**
   - Base URL: `http://localhost:8000` or `http://localhost:11000`
   - Ensure your services are running on the correct ports

3. **Run Tests in Order:**
   The tests are designed to run sequentially as they depend on each other:
   
   a. **Ping Main Service** - Verify service is running
   b. **Add Bucket** - Creates a bucket and stores bucketId for later tests
   c. **Get All Buckets** - Verifies bucket creation
   d. **Add Lead - Image Processing with Agent** - Upload an image for AI processing
   e. **Get All Leads** - Verify leads were extracted and stored
   f. **Get Leads by Bucket ID** - Filter leads by specific bucket
   g. **Update Lead Status** - Test status updates using dynamically captured lead ID
   h. **Update Lead Notes** - Test notes updates using dynamically captured lead ID

### Test Details

#### Add Lead - Image Processing with Agent

This is the main test for the new functionality:

- **Method:** POST
- **Endpoint:** `/api/main-service/leads/add-lead`
- **Parameters:**
  - `file`: Upload an image file (business card, LinkedIn profile, etc.)
  - `bucket_id`: Target bucket ID (uses variable from "Add Bucket" test)

**Expected Responses:**
- **201 Success:** Image processed successfully, lead information extracted and stored
- **500 Error:** Processing failed (could be due to no lead info in image, API issues, etc.)

**Test Validation:**
- Verifies response structure
- Checks that bucket_id matches request
- Validates agent processing result
- Logs detailed processing information

#### Sample Test Images

Use the provided script to create test images:

```bash
python create_test_images.py
```

This creates:
- `sample_business_card.png` - Business card with contact details
- `sample_linkedin_profile.png` - LinkedIn profile screenshot  
- `sample_contact_info.png` - Simple contact information

## Agent Behavior

The AI agent will:

1. **Analyze the Image:** Use GPT-4 Vision to examine the uploaded image
2. **Identify Lead Information:** Look for names, contact details, social profiles, etc.
3. **Extract Data:** Pull out relevant information in structured format
4. **Make Decision:** Choose to either:
   - Do nothing (if no lead information found)
   - Call the `add_lead_to_crm` tool to store the lead

## Expected Test Results

### Successful Image Processing (201 Response)
```json
{
  "message": "Image processed successfully",
  "agent_result": "Successfully added lead to CRM. Lead ID: abc-123-def...",
  "bucket_id": "bucket-uuid-here",
  "filename": "uploaded_image.png"
}
```

### No Lead Information Found (201 Response)
```json
{
  "message": "Image processed successfully", 
  "agent_result": "No lead information was found in this image.",
  "bucket_id": "bucket-uuid-here",
  "filename": "uploaded_image.png"
}
```

### Processing Error (500 Response)
```json
{
  "message": "Error processing image",
  "error": "Error details here",
  "agent_result": "Error processing image: ..."
}
```

## Troubleshooting

### Common Issues

1. **503 Service Unavailable:**
   - Check if MongoDB service is running on port 10000
   - Verify `MONGODB_SERVICE` environment variable

2. **Agent Not Available:**
   - Ensure `OPENAI_API_KEY` is set
   - Check OpenAI API quota and billing

3. **422 Validation Error:**
   - Make sure to upload a file in the request
   - Check file format (PNG, JPG supported)

4. **Agent Processing Errors:**
   - Try with different images
   - Check console logs for detailed error messages
   - Verify image contains readable text/contact information

### Debug Information

The tests include extensive logging:
- Pre-request scripts show configuration
- Test scripts log processing results
- Console output shows detailed agent responses

## Integration with MongoDB Service

The AI agent integrates seamlessly with the existing MongoDB service:

1. **Lead Storage:** Extracted leads are stored using the existing `/api/mongodb-service/leads/add-lead` endpoint
2. **Bucket Association:** Leads are automatically associated with the specified bucket
3. **Data Validation:** All extracted data follows the existing lead schema
4. **Status Management:** New leads default to "Cold Message" status

## Performance Considerations

- Image processing can take 10-30 seconds depending on image complexity
- OpenAI API rate limits may apply
- Large images may need to be resized for optimal processing
- The agent includes timeout handling for robustness

## Security Notes

- Images are temporarily stored during processing and then deleted
- OpenAI API key should be kept secure
- Image data is sent to OpenAI for processing (consider privacy implications)
- No persistent storage of uploaded images on the server