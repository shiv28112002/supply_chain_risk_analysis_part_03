# Part 3 — GenAI Text Analytics

## AI Supply Chain Risk Intelligence Platform

### Project Area
Customer Support Ticket Sentiment and Aspect Analysis

---

## 1. Overview

Part 3 of my capstone project focuses on using Generative AI for
customer support text analytics.

The main objective is to use prompt engineering and the Gemini API to
classify customer support tickets and return structured outputs that can
be used for further analytics.

The workflow includes:

1. Loading and inspecting customer support ticket data.
2. Identifying the free-text ticket field.
3. Connecting Python to the Gemini API.
4. Creating a reusable LLM API wrapper.
5. Adding retry handling for failed API calls.
6. Designing Zero-Shot, Few-Shot, and Role-Prompted prompts.
7. Running a controlled 15-call prompt experiment.
8. Parsing and validating JSON responses.
9. Comparing prompting strategies based on output consistency.
10. Extending the best-performing prompt to aspect-based sentiment
    analysis.

---

# 2. Dataset

The dataset used in this part contains customer support tickets from an
e-commerce environment.

### Dataset characteristics

- Records: 500
- Columns: 17
- Main text field: `Ticket Description`

Important fields used in the prompt include:

| Column | Purpose |
|---|---|
| `Ticket Type` | Type/category of support request |
| `Ticket Subject` | Short description of the issue |
| `Ticket Description` | Main customer free-text input |
| `Product Purchased` | Product associated with the ticket |
| `Ticket Priority` | Priority assigned to the ticket |
| `Ticket Channel` | Channel through which the ticket was received |

The free-text field was identified as:

`Ticket Description`

This field is used as the primary evidence for sentiment analysis.

---

# 3. Technology Stack

- Python
- Pandas
- Google Gemini API
- JSON
- Python logging
- Jupyter Notebook

---

# 4. GenAI Text Analytics Pipeline

## 4.1 Load and Inspect the Dataset

The customer support ticket dataset was loaded into a Pandas DataFrame.

The dataset was inspected to understand:

- Number of rows and columns
- Available ticket fields
- Data types
- Missing values
- Available text fields

The main free-text field required for the LLM classification task was
identified as `Ticket Description`.

---

## 4.2 Identify the Feedback Text Column

The pipeline checks possible text-column names and identifies the
available customer feedback field.

For this dataset, the selected field is:

```text
Ticket Description
```

This column contains the actual customer support message that is passed
to Gemini.

---

## 4.3 Prompt Templates

Three prompt-engineering strategies were implemented.

### Zero-Shot Prompt

The Zero-Shot prompt directly instructs the model to classify the
customer ticket sentiment without providing classification examples.

The allowed sentiment labels are:

- `positive`
- `negative`
- `neutral`

The model is also asked to return:

- sentiment label
- confidence level
- concise reason

### Few-Shot Prompt

The Few-Shot prompt provides examples of positive, negative, and neutral
customer tickets before asking the model to classify a new ticket.

The examples help anchor both the classification task and the expected
JSON output format.

### Role-Prompted Prompt

The Role-Prompted strategy assigns Gemini the role of a senior
customer-insights analyst specializing in e-commerce customer support
analytics.

The model is instructed to use the customer's actual wording as the
primary evidence for sentiment.

---

# 5. Structured JSON Output

All three prompting strategies use the same locked output structure:

```json
{
  "label": "positive|negative|neutral",
  "confidence": "low|medium|high",
  "reason": "string"
}
```

The locked schema is used so that the generated responses can be parsed
and validated programmatically.

The prompt explicitly instructs the model to:

- return JSON only
- use the required field names
- use only the allowed sentiment labels
- use only the allowed confidence levels
- provide a concise reason
- avoid Markdown and additional fields

Structured output is important because free-form LLM responses are
difficult to use reliably in downstream analytics pipelines.

---

# 6. Gemini API Integration

A reusable `call_llm()` function was created to send prompts to the
Gemini API.

The function accepts:

```text
prompt
temperature
max_tokens
```

Example:

```python
response = call_llm(
    prompt=prompt,
    temperature=0.2,
    max_tokens=500
)
```

A basic API connectivity test was performed before running the full
experiment.

The test confirmed that the Gemini API connection and reusable wrapper
were working.

---

# 7. Retry Handling

A retry-enabled wrapper was implemented around the Gemini API call.

If a request fails, the system retries the request up to three times.

The process also logs:

- successful attempts
- failed attempts
- final failures

This prevents one API failure from stopping the complete text analytics
pipeline.

Example:

```python
response = call_llm_with_retry(
    prompt=prompt,
    temperature=0.2,
    max_tokens=500
)
```

---

# 8. Prompt Experiment

To compare the three prompting strategies fairly, the same five test
records were used for each strategy.

Therefore:

```text
3 prompting strategies × 5 records = 15 attempted calls
```

The experiment included:

- Zero-Shot → 5 records
- Few-Shot → 5 records
- Role-Prompted → 5 records

The same output schema was used for all three strategies.

This makes the experiment a controlled comparison of prompting
approaches.

---

# 9. JSON Parsing and Schema Validation

After receiving the Gemini responses, the responses were processed using
Python's `json.loads()`.

Each response was checked for:

### JSON validity

Whether the response could be parsed as valid JSON.

### Required fields

The response was expected to contain exactly:

```text
label
confidence
reason
```

### Label validation

The allowed labels were:

```text
positive
negative
neutral
```

### Confidence validation

The allowed confidence levels were:

```text
low
medium
high
```

### Reason validation

The `reason` field was required to be a string.

API failures and JSON parsing failures were logged separately so that a
single failed response did not crash the pipeline.

---

# 10. 15-Call Experiment Results

The recorded experiment produced the following results:

| Metric | Result |
|---|---:|
| Total calls processed | 15 |
| Successful API responses | 12 |
| API failures | 3 |
| Valid JSON responses | 0 |
| Schema-conformant responses | 0 |

The successful API responses were received, but several responses were
truncated before the JSON object was completed.

For example, some responses stopped after fields such as:

```json
{
  "label": "negative",
  "confidence": "high",
```

without completing the `reason` field and closing the JSON object.

Because the response was incomplete, `json.loads()` correctly rejected
it.

The generated responses were not manually modified to create artificial
successful results.

---

# 11. Comparison of Prompting Strategies

The intended comparison is based on the proportion of responses that are
both:

1. Valid JSON
2. Schema-conformant

The comparison is performed using the validated results:

```python
results_df = pd.DataFrame(validated_results)

comparison = (
    results_df
    .groupby("template")
    .agg(
        total_calls=("record_number", "count"),
        valid_json=("valid_json", "sum"),
        schema_conformant=("schema_conformant", "sum")
    )
    .reset_index()
)

comparison["invalid_json"] = (
    comparison["total_calls"]
    - comparison["valid_json"]
)

comparison["schema_nonconformant"] = (
    comparison["total_calls"]
    - comparison["schema_conformant"]
)

comparison["schema_conformance_rate_percent"] = (
    comparison["schema_conformant"]
    / comparison["total_calls"]
    * 100
)

comparison = comparison.sort_values(
    "schema_conformance_rate_percent",
    ascending=False
)

display(comparison)
```

Because the recorded execution produced zero valid JSON responses, a
meaningful prompt winner could not be established from this run.

No template is falsely declared the winner.

---

# 12. Winner Selection

The intended selection rule is:

> The prompting strategy with the highest schema-conformant JSON rate is
> selected as the most consistently valid template.

The notebook contains the winner-selection logic, but the recorded
execution did not produce a schema-conformant response for any template.

Therefore, the current experiment does not claim a winner.

A future execution with sufficient API availability can repeat the same
15-call experiment and select the winner from the actual validation
results.

---

# 13. Aspect-Based Sentiment Analysis Extension

## Objective

The next extension is aspect-based sentiment analysis.

Overall sentiment gives only one view of a customer ticket. A customer
can be dissatisfied with a product but satisfied with the support
experience.

Aspect-based sentiment analysis separates sentiment into specific
business dimensions.

For this e-commerce support domain, two relevant aspects are:

- `product`
- `service`

The task requires at least two aspects per record.

---

## 13.1 Aspect-Based JSON Structure

The extension uses a structured JSON format similar to the approach used
for JSON-constrained sentiment classification:

```json
{
  "product": {
    "sentiment": "positive|negative|neutral|not_mentioned",
    "actionable_phrase": "3-6 words"
  },
  "service": {
    "sentiment": "positive|negative|neutral|not_mentioned",
    "actionable_phrase": "3-6 words"
  }
}
```

If an aspect is not mentioned in the ticket, it should be returned as
`not_mentioned` rather than inventing sentiment.

The actionable phrase should be short and describe what the customer
liked or disliked.

Examples:

```text
"Improve product reliability"
"Faster support response"
"Easy product setup"
"Resolve refund delays"
```

---

# 14. Aspect-Based Prompt

The aspect-based extension is designed to follow the same prompt
engineering principles used in the earlier experiment.

The prompt should:

1. Assign the model the role used by the selected prompting strategy.
2. Define the two required aspects.
3. Define the allowed sentiment labels.
4. Define the `not_mentioned` condition.
5. Require a 3–6 word actionable phrase.
6. Require JSON-only output.
7. Prohibit invented information.

Example structure:

```text
Act as a senior customer-insights analyst specializing in e-commerce
customer support analytics.

Analyze the following customer support ticket for two aspects:

1. product
2. service

For each aspect:
- classify sentiment as positive, negative, neutral, or not_mentioned
- provide one actionable phrase containing 3–6 words
- base the result only on information stated in the ticket

Return ONLY valid JSON using exactly this structure:

{
  "product": {
    "sentiment": "positive|negative|neutral|not_mentioned",
    "actionable_phrase": "3-6 words"
  },
  "service": {
    "sentiment": "positive|negative|neutral|not_mentioned",
    "actionable_phrase": "3-6 words"
  }
}
```

This follows the study-material approach of using structured JSON for
aspect-level sentiment and adding short actionable phrases for each
aspect.

---

# 15. Ten-Record Aspect Analysis

The final extension is designed to run on a subset of 10 customer
support ticket records.

Required output per record:

- Product sentiment
- Product actionable phrase
- Service sentiment
- Service actionable phrase

The final README table will have the following structure:

| Record | Product Sentiment | Product Actionable Phrase | Service Sentiment | Service Actionable Phrase |
|---:|---|---|---|---|
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 3 | — | — | — | — |
| 4 | — | — | — | — |
| 5 | — | — | — | — |
| 6 | — | — | — | — |
| 7 | — | — | — | — |
| 8 | — | — | — | — |
| 9 | — | — | — | — |
| 10 | — | — | — | — |

---

# 16. Error Handling and Reliability

The project does not assume that an LLM response is automatically
usable.

The pipeline therefore checks:

```text
API response
     ↓
JSON parsing
     ↓
Schema validation
     ↓
Structured analytical result
```

This is important because a model can return a response that is
semantically reasonable but still unusable for a downstream program if
the expected structure is not followed.

The experiment demonstrated this issue directly through truncated JSON
responses.

---

# 17. Limitations

The main limitation during the recorded execution was Gemini API
availability and incomplete structured responses.

The experiment recorded actual API behaviour instead of manually
creating successful outputs.

As a result:

- the 15 calls were attempted;
- some API calls failed;
- some responses were incomplete;
- no response passed the complete JSON/schema validation in the
  recorded run;
- a statistically meaningful winning prompt could not be established.

The aspect-based extension is defined in the notebook workflow, but the
final 10-record generated table requires successful API responses.

---

# 18. Future Improvements

The following improvements can be applied when API availability allows:

- Re-run the 15-call prompt comparison.
- Use the actual schema-conformance rate to select the winner.
- Run the aspect-based extension on 10 records.
- Store the aspect results as structured JSON.
- Add the 10-record aspect table to this README.
- Analyze common product and service issues.
- Use the extracted phrases to identify actionable customer-support
  improvements.

---

# 19. Project Structure

```text
Part_3_GenAI_Text_Analytics/
│
├── genai_text_analytics.ipynb
├── README.md
└── data/
    └── customer_support_tickets.csv
```

---

# 20. Conclusion

Part 3 demonstrates the use of Generative AI for customer support text
analytics using prompt engineering and the Gemini API.

The implementation covers:

- customer support text preparation
- free-text column identification
- reusable Gemini API integration
- retry handling
- Zero-Shot prompting
- Few-Shot prompting
- Role-Based prompting

The experiment also demonstrates an important practical lesson in
GenAI applications: requesting structured output is not enough by
itself. Generated responses must be parsed and validated before they
are used in an automated analytics pipeline.

The recorded API limitation and incomplete JSON responses are retained
as part of the experiment results rather than being manually changed.
