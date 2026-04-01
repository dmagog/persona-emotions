# Trait Data Generation using Claude API

This module uses the Anthropic Claude API to generate trait evaluation data including instruction pairs, questions, and evaluation prompts for different personality traits.

## Setup

1. Install the required dependencies:
```bash
pip install -r ../requirements.txt
```

2. Set your Anthropic API key as an environment variable:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

### Command Line Interface

Generate data for all traits defined in `trait_definitions.json`:
```bash
python generate.py
```

Generate data for specific traits:
```bash
python generate.py --traits evil sycophantic humorous
```

Specify output directory:
```bash
python generate.py --output-dir custom_output_folder
```

Use a different Claude model:
```bash
python generate.py --model claude-3-5-sonnet-20241022
```

Add delay between API calls:
```bash
python generate.py --delay 2.0
```

### Programmatic Usage

```python
from generate import ClaudeDataGenerator

# Initialize generator
generator = ClaudeDataGenerator()

# Generate data for a trait
data = generator.generate_trait_data(
    trait="cooperative",
    trait_instruction="Description of cooperative behavior...",
    question_instruction="Focus on team scenarios..."
)

# Save the data
generator.save_trait_data("cooperative", data)
```

## Output Format

The generated data follows this structure:

```json
{
    "instruction": [
        {
            "pos": "Positive instruction that encourages the trait",
            "neg": "Negative instruction that discourages the trait"
        }
        // ... 5 instruction pairs total
    ],
    "questions": [
        "Question 1 that could elicit the trait",
        "Question 2 that could elicit the trait",
        // ... 40 questions total
    ],
    "eval_prompt": "Evaluation prompt for scoring trait presence in responses"
}
```

## Configuration

### Trait Definitions

Traits are defined in `trait_definitions.json`:

```json
{
    "trait_name": {
        "instruction": "Description of the trait behavior",
        "question_instruction": "Additional guidance for question generation"
    }
}
```

### Available Traits

The default traits include:
- evil
- sycophantic 
- hallucinating
- humorous
- impolite
- optimistic
- apathetic

## Error Handling

The script includes:
- Retry logic for API calls with exponential backoff
- Validation of generated JSON structure
- Error logging and graceful failure handling
- Rate limiting with configurable delays

## API Rate Limits

To respect Claude API rate limits:
- Default delay of 1 second between requests
- Exponential backoff retry logic
- Configurable delay via `--delay` parameter
