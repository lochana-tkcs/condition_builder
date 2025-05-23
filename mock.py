import streamlit as st
import pandas as pd
from openai import OpenAI
import regex as re
import json
from datetime import datetime
from dateutil import parser, relativedelta

# Set your OpenAI API key
api_key = st.secrets["openai_api_key"]

# Initialize the OpenAI client with the API key
client = OpenAI(
    api_key=api_key)

condition_schema = {
  "type": "json_schema",
  "json_schema": {
    "name": "generate_condition",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "condition": {
          "type": "object",
          "description": "A logical condition with an operator and conditions.",
          "properties": {
            "Operator": {
              "type": "string",
              "enum": ["OR", "AND"],
              "description": "The logical operator combining the conditions."
            },
            "Conditions": {
              "type": "array",
              "description": "Array of conditions or condition groups.",
              "items": {
                "anyOf": [
                  { "$ref": "#/$defs/condition" },
                  { "$ref": "#/$defs/condition_group" }
                ]
              }
            }
          },
          "required": ["Operator", "Conditions"],
          "additionalProperties": False
        }
      },
      "required": ["condition"],
      "additionalProperties": False,
      "$defs": {
        "condition": {
          "type": "object",
          "description": "A single condition specifying a column and operator.",
          "anyOf": [
            {
              "description": "Condition for Text or Numeric columns",
              "properties": {
                "Column_Name": {
                  "type": "string",
                  "description": "The name of the column to apply the condition to."
                },
                "Column_Type": {
                  "type": "string",
                  "enum": ["Text", "Numeric"],
                  "description": "Specifies the data type of the column: Text or Numeric."
                },
                "Column_Operator": {
                  "type": "string",
                  "enum": [
                    "is",
                    "is NOT",
                    "is less than",
                    "is less than or equal to",
                    "is greater than",
                    "is greater than or equal to",
                    "is MAX value",
                    "is NOT MAX value",
                    "is MIN value",
                    "is NOT MIN value",
                    "is Empty",
                    "is NOT Empty",
                    "contains",
                    "does NOT contain",
                    "starts with",
                    "ends with",
                    "does NOT start with",
                    "does NOT end with"
                  ],
                  "description": """
                    The operands compatible with Text Columns - is, is NOT, contains, does NOT contain, starts with, ends with, does NOT start with, does NOT end with, is EMPTY, is NOT EMPTY.
                    The operands compatible with Numerical Columns - is, is NOT, is less than, is less than or equal to, is greater than, is greater than or equal to, is MAX value, is NOT MAX value, is MIN value, is NOT MIN value, is EMPTY, is NOT EMPTY.
                  """
                },
                "Operand_Type": {
                  "type": "string",
                  "enum": ["Value", "Column Value"]
                },
                "Operand": {
                  "type": "array",
                  "description": "The value(s) to compare against.",
                  "items": {
                    "anyOf": [
                      { "type": "string" },
                      { "type": "number" },
                      { "type": "boolean" }
                    ]
                  }
                }
              },
              "required": ["Column_Name", "Column_Type", "Column_Operator", "Operand_Type", "Operand"],
              "additionalProperties": False
            },
            {
              "description": "Condition for Date/Time columns with Date format",
              "properties": {
                "Column_Name": {
                  "type": "string",
                  "description": "The name of the column to apply the condition to."
                },
                "Column_Type": {
                  "type": "string",
                  "enum": ["Date/Time"],
                  "description": "Specifies the data type of the column: Date/Time."
                },
                "Column_Operator": {
                  "type": "string",
                  "enum": [
                    "is",
                    "is NOT",
                    "is earlier than",
                    "is on or earlier than",
                    "is later than",
                    "is on or later than",
                    "is Empty",
                    "is NOT Empty"
                  ],
                  "description": "The operands compatible with Date/Time Columns - is, is NOT, is earlier than, is on or earlier than, is later than, is on or later than, is EMPTY, is NOT EMPTY."
                },
                "Operand_Type": {
                  "type": "string",
                  "enum": ["Value", "Column Value"]
                },
                "Date_Format": {
                  "type": "string",
                  "enum": ["Date", "Date-Time in seconds", "Date/Minute", "Date/Hour", "Year/Month", "Year", "Day of Month", "Month of Year", "Earliest single value", "Latest single value", "Weekday"],
                  "description": "Month is used to filter out records in all years"
                },
                "Operand": {
                  "type": "array",
                  "description": "The value(s) to compare against, must be strings for Date or ISO Date-Time formats.",
                  "items": { "type": "string" }
                }
              },
              "required": ["Column_Name", "Column_Type", "Column_Operator", "Operand_Type", "Date_Format", "Operand"],
              "additionalProperties": False
            },
            {
              "description": "Condition for Date/Time columns with are x from their earliest or latest date",
              "properties": {
                "Column_Name": {
                  "type": "string",
                  "description": "The name of the column to apply the condition to."
                },
                "Column_Type": {
                  "type": "string",
                  "enum": ["Date/Time"],
                  "description": "Specifies the data type of the column: Date/Time."
                },
                "Column_Operator": {
                  "type": "string",
                  "enum": [
                    "is",
                    "is NOT",
                    "is earlier than",
                    "is on or earlier than",
                    "is later than",
                    "is on or later than"
                  ],
                  "description": "The operands compatible with Date/Time Columns - is, is NOT, is earlier than, is on or earlier than, is later than, is on or later than, is EMPTY, is NOT EMPTY."
                },
                "Operand_Type": {
                  "type": "string",
                  "enum": ["Value"]
                },
                "Date_Format": {
                  "type": "string",
                  "enum": ["Earliest single value", "Latest single value", "Current Date (UTC)"],
                  "description": "The reference point for comparison. Determines whether to calculate the offset from the earliest or latest timestamp in the column."
                },
                "Operand": {
                  "type": "array",
                  "description": "The offset value(s) to apply to the reference point. For example, 4 means '4 units after the reference date'. Accepts both positive and negative numbers. For example, -3 with unit 'Day' means 3 days before the reference date.",
                  "items": { "type": "string" }
                },
                "Unit": {
                  "type": "string",
                  "enum": ["Year", "Month", "Day", "Hour", "Minute", "Second", "Week"],
                  "description": "The unit of time used to interpret the offset (e.g., '4' with 'Hour' means 4 hours after the reference date)."
                }
              },
              "required": ["Column_Name", "Column_Type", "Column_Operator", "Operand_Type", "Date_Format", "Operand", "Unit"],
              "additionalProperties": False
            }
          ]
        },
        "condition_group": {
          "type": "object",
          "description": "A group of conditions with a logical operator.",
          "properties": {
            "Operator": {
              "type": "string",
              "enum": ["OR", "AND"],
              "description": "The logical operator combining the conditions."
            },
            "Conditions": {
              "type": "array",
              "description": "Array of conditions or condition groups.",
              "items": {
                "anyOf": [
                  { "$ref": "#/$defs/condition" },
                  { "$ref": "#/$defs/condition_group" }
                ]
              }
            }
          },
          "required": ["Operator", "Conditions"],
          "additionalProperties": False
        }
      }
    }
  }
}

def transform_condition(raw_condition):
    """Transform API response to desired format."""
    operator_map = {
        "is": "IN_LIST", "is NOT": "NOT_IN_LIST", "is less than": "LT",
        "is less than or equal to": "LTE", "is greater than": "GT",
        "is greater than or equal to": "GTE", "is MAX value": "IS_MAXVAL",
        "is NOT MAX value": "IS_NOT_MAXVAL", "is MIN value": "IS_MINVAL",
        "is NOT MIN value": "IS_NOT_MINVAL", "is Empty": "IS_EMPTY",
        "is NOT Empty": "IS_NOT_EMPTY", "contains": "CONTAINS",
        "does NOT contain": "NOT_CONTAINS", "starts with": "STARTS_WITH",
        "ends with": "ENDS_WITH", "does NOT start with": "NOT_STARTS_WITH",
        "does NOT end with": "NOT_ENDS_WITH", "is earlier than": "LT",
        "is on or earlier than": "LTE", "is later than": "GT",
        "is on or later than": "GTE",
    }

    date_format_mapping = {
        "Date": {"field": "TRUNCATE", "value": "DAY", "format": lambda x: parser.parse(x).strftime("%Y-%m-%d") if x else None},
        "Date-Time in seconds": {"field": "TRUNCATE", "value": "SECOND", "format": lambda x: parser.parse(x).strftime("%Y-%m-%d %H:%M:%S") if x else None},
        "Date/Minute": {"field": "TRUNCATE", "value": "MINUTE", "format": lambda x: parser.parse(x).strftime("%Y-%m-%d %H:%M") if x else None},
        "Date/Hour": {"field": "TRUNCATE", "value": "HOUR", "format": lambda x: parser.parse(x).strftime("%Y-%m-%d %H") if x else None},
        "Year/Month": {"field": "TRUNCATE", "value": "MONTH", "format": lambda x: parser.parse(x).strftime("%Y-%m-01") if x else None},
        "Year": {"field": "TRUNCATE", "value": "YEAR", "format": lambda x: parser.parse(x).strftime("%Y-01-01") if x else None},
        "Day of Month": {"field": "COMPONENT", "value": "day", "format": lambda x: int(x) if x.isdigit() else int(parser.parse(x).strftime("%d"))},
        "Month of Year": {"field": "COMPONENT", "value": "month_text", "format": lambda x: x if x in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"] else parser.parse(x).strftime("%B")},
        "Earliest single value": {"field": "FUNCTION", "value": "MIN", "format": lambda x: None},
        "Latest single value": {"field": "FUNCTION", "value": "MAX", "format": lambda x: None},
        "Current Date (UTC)": {"field": "FUNCTION", "value": "SYSTEM_DATE", "format": lambda x: None},
        "Weekday": {"field": "COMPONENT", "value": "weekday_text", "format": lambda x: x if x in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] else parser.parse(x).strftime("%A")},
    }

    def parse_relative_time(relative_str, base_date):
        """Parse relative time strings like '1 year' or '3 months'."""
        if not relative_str:
            return None
        try:
            num, unit = relative_str.split()
            num = int(num)
            unit = unit.lower()
            delta = relativedelta(years=num) if unit.startswith("year") else relativedelta(months=num) if unit.startswith("month") else None
            return (base_date - delta).strftime("%Y-%m-%d %H:%M:%S") if delta else None
        except (ValueError, AttributeError):
            return None

    def get_mapped_operator(column_op, column_type):
        """Map operator, adjusting for Date/Time types."""
        mapped_op = operator_map.get(column_op)
        if column_type == "Date/Time":
            return "EQ" if column_op == "is" else "NE" if column_op == "is NOT" else mapped_op
        return mapped_op

    def process_operand(operand, operand_type, column_op):
        """Process operand based on type and operator."""
        if not operand:
            return None
        if operand_type == "Column Value" and column_op in ["is", "is one of"]:
            return operand[0]
        if column_op in ["is", "is NOT", "contains", "does NOT contain", "is one of"]:
            return operand
        return operand[0]

    def transform_single_condition(cond, base_date):
        """Transform a single condition."""
        column_name = cond["Column_Name"]
        column_op = cond["Column_Operator"]
        operand = cond.get("Operand")
        operand_type = cond["Operand_Type"]
        date_format = cond.get("Date_Format")
        column_type = cond.get("Column_Type")
        unit = cond.get("Unit")
        mapped_op = get_mapped_operator(column_op, column_type)
        output_column_name = "column_2" if column_name == "Time" else column_name

        # Handle empty operand validation
        special_formats = ["Earliest single value", "Latest single value", "Current Date (UTC)"]
        special_ops = ["IS_EMPTY", "IS_NOT_EMPTY", "IS_MAXVAL", "IS_NOT_MAXVAL", "IS_MINVAL", "IS_NOT_MINVAL"]
        if not operand and date_format not in special_formats and mapped_op not in special_ops:
            raise ValueError(f"Operand cannot be empty for condition on {column_name}")

        # Handle Date/Time operand
        if column_type == "Date/Time" and operand:
            operand = operand.replace('T', ' ') if 'T' in operand else operand

        # Handle special operators
        if mapped_op in special_ops:
            return {output_column_name: {mapped_op: True}}

        # Handle date format mapping
        if date_format in date_format_mapping:
            mapping = date_format_mapping[date_format]
            try:
                if mapping["field"] == "FUNCTION" and not unit or mapped_op in ["IS_EMPTY", "IS_NOT_EMPTY"]:
                    return {
                        output_column_name: {
                            mapped_op: {"VALUE": {mapping["field"]: mapping["value"]} if mapping["field"] == "FUNCTION" else True}
                        }
                    }
                value = operand[0] if operand else None
                if date_format in special_formats and value:
                    if unit:
                        return {
                            output_column_name: {
                                mapped_op: {
                                    "VALUE": {
                                        "FUNCTION": mapping["value"],
                                        "DELTA": {unit.upper(): int(value)}
                                    }
                                }
                            }
                        }
                    value = parse_relative_time(value, base_date)
                formatted_value = mapping["format"](value) if value else None
                return {
                    output_column_name: {
                        mapped_op: {
                            "VALUE": {
                                mapping["field"]: mapping["value"],
                                "VALUE": formatted_value
                            }
                        }
                    }
                }
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid date format for {value or 'None'} in {date_format} condition: {str(e)}")

        # Handle standard conditions
        key = "COLUMN" if operand_type == "Column Value" else "VALUE"
        value = process_operand(operand, operand_type, column_op)
        return {output_column_name: {mapped_op: {key: value}}}

    # Main processing
    base_date = datetime(2025, 5, 20, 18, 45)  # May 20, 2025, 06:45 PM IST

    if "Operator" not in raw_condition:
        return transform_single_condition(raw_condition, base_date)

    # Handle nested conditions
    logical_op = raw_condition["Operator"]
    result = {logical_op: []}
    for cond in raw_condition["Conditions"]:
        result[logical_op].append(transform_condition(cond))
    return result

def simplify_conditions(condition):
    # If this is a logical block with 'Operator' and 'Conditions'
    if isinstance(condition, dict) and "Operator" in condition and "Conditions" in condition:
        simplified_conditions = [simplify_conditions(cond) for cond in condition["Conditions"]]

        # If there's only one condition, return it directly (unwrap)
        if len(simplified_conditions) == 1:
            return simplified_conditions[0]
        else:
            # Otherwise, return the logical block with simplified conditions
            return {
                "Operator": condition["Operator"],
                "Conditions": simplified_conditions
            }

    # If it's already a leaf condition, return as-is
    return condition

def get_column_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "Date/Time"
    else:
        try:
            pd.to_datetime(series.dropna().iloc[0])  # Try converting first non-null value
            return "Date/Time"
        except Exception:
            return "Text"

def fun3(data, prompt):
    for col in data.columns:
        column_values = [str(row) for row in data[col].head(20)]
        all_values_str = ", ".join(column_values)

        # Infer column type
        col_type = get_column_type(data[col])

        col_clean = re.sub(r"\(.*?\)", "", col).strip()

        # Add updated type
        col_header = f"Column: {col_clean}({col_type})"

        prompt += f"\n{col_header}\n"
        prompt += "Values:\n" + all_values_str + "\n"

    prompt += "\nGiven the intent, output just the dictionary and no other text"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        response_format = condition_schema,
        temperature=0
    )

    # output = response.choices[0].message.content.strip()

    raw_condition = json.loads(response.choices[0].message.content)["condition"]
    raw_condition = simplify_conditions(raw_condition)
    condition_output = {"condition": transform_condition(raw_condition)}
    # print("Generated Condition:", json.dumps(condition_output, indent=2))
    return raw_condition, condition_output

# Recursive renderer for displaying condition_dict
def render_conditions(data, level=0):
    indent = level * 20
    if "Operator" in data and "Conditions" in data:
        st.markdown(
            f"<div style='margin-left:{indent}px; padding: 10px; border: 1px solid #ccc; border-radius: 8px;'>"
            f"<b>{data['Operator']}</b> Block",
            unsafe_allow_html=True
        )
        for cond in data["Conditions"]:
            render_conditions(cond, level + 1)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        column_name = data.get('Column_Name', '')
        column_type = data.get('Column_Type', '')
        column_operator = data.get('Column_Operator', '')
        operand_type = data.get('Operand_Type', '')
        operand = ', '.join(map(str, data.get('Operand', [])))
        date_format = data.get('Date_Format', '')
        unit = data.get('Unit', '')

        parts = [
            f"<b>{column_name}</b>",
            f"<span style='color: gray;'>({column_type})</span>",
            f"'{column_operator}'",
            f"<i>{operand}</i>",
            f"<span style='color: gray;'>({operand_type})</span>"
        ]
        if date_format:
            parts.append(f"<i>{date_format}</i> <span style='color: gray;'>(Value Type)</span>")
        if unit:
            parts.append(f"<i>{unit}</i> <span style='color: gray;'>(Unit)</span>")

        line = " ".join(parts)

        st.markdown(
            f"<div style='margin-left:{indent}px; padding: 5px 0;'>{line}</div>",
            unsafe_allow_html=True
        )

# Streamlit App
st.title("Dataset Condition Builder")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [
        f"{col}(num)" if pd.api.types.is_numeric_dtype(df[col]) else f"{col}(text)"
        for col in df.columns
    ]

    st.subheader("🔍 Data Preview")
    st.dataframe(df)

    st.subheader("Enter your condition intent")
    user_intent = st.text_input("What condition do you want to apply?")

    if user_intent:
        prompt_template = f"""
        You are an intelligent assistant designed to convert natural language queries into structured condition logic for filtering datasets.

        You are given 20 sample rows of a dataset. Based on these rows, you understand the column names, value types, and typical entries.

        Your task is to analyze the user’s intent and output a **nested condition structure** in the form of a **dictionary**.

        User intent: {user_intent}
        The columns of the dataset are as follows:
        {', '.join(df.columns)}
        """

        # Call the function to get the structured condition dictionary
        raw_condition, condition_dict = fun3(df, prompt_template)

        st.subheader("🧾 Condition Output")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Visual Representation")
            render_conditions(raw_condition)

        with col2:
            st.markdown("### JSON Dictionary")
            st.code(json.dumps(condition_dict["condition"], indent=2), language="json")
