import streamlit as st
import pandas as pd
from openai import OpenAI
import regex as re
import random
import json

# Set your OpenAI API key
api_key = st.secrets["openai_api_key"]

# Initialize the OpenAI client with the API key
client = OpenAI(
    api_key=api_key)

#working p
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
          "properties": {
            "Column_Name": {
              "type": "string",
              "description": "The name of the column to apply the condition to."
            },
            "Column_Type": {
              "type": "string",
              "enum": ["Text", "Numeric", "Date/Time"],
              "description": "It can Date/Time, Numeric or Text. Specifies the data type of the column on which the condition will be applied"
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
                              The operands compatible for Text Columns - is, is NOT, contains, does NOT contain, starts with, ends with, does NOT start with, does NOT end with, is EMPTY, is NOT EMPTY.
                              The operands compatible for Numerical Columns - is, is NOT, is less than, is less than or equal to, is greater than, is greater than or equal to, is MAX value, is NOT MAX value, is MIN value, is NOT MIN value, is EMPTY, is NOT EMPTY.
                             """
            },
            "Operand_Type": {
                      "type": "string",
                      "enum": ["Value", "Column Value"]
                    },
            "Operand": {
              "type": "array",
              "description": "The value(s) to compare against. The list can have more than one item",
              "items": {
                "anyOf": [
                  {"type": "string"},
                  {"type": "number"},
                  {"type": "boolean"}
                ]
              }
            }
          },
          "required": ["Column_Name", "Column_Type", "Column_Operator", "Operand_Type", "Operand"],
          "additionalProperties": False
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
        "is": "IN_LIST",
        "is NOT": "NOT_IN_LIST",
        "is less than": "LT",
        "is less than or equal to": "LTE",
        "is greater than": "GT",
        "is greater than or equal to": "GTE",
        "is MAX value": "IS_MAXVAL",
        "is NOT MAX value": "IS_NOT_MAXVAL",
        "is MIN value": "IS_MINVAL",
        "is NOT MIN value": "IS_NOT_MINVAL",
        "is Empty": "IS_EMPTY",
        "is NOT Empty": "IS_NOT_EMPTY",
        "contains": "CONTAINS",
        "does NOT contain": "NOT_CONTAINS",
        "starts with": "STARTS_WITH",
        "ends with": "ENDS_WITH",
        "does NOT start with": "NOT_STARTS_WITH",
        "does NOT end with": "NOT_ENDS_WITH",
    }

    if "Operator" not in raw_condition:
        column_name = raw_condition["Column_Name"]
        column_op = raw_condition["Column_Operator"]
        operand = raw_condition["Operand"]
        operand_type = raw_condition["Operand_Type"]

        if not operand:
            raise ValueError(f"Operand cannot be empty for condition on {column_name}")

        key = "COLUMN" if operand_type == "Column Value" else "VALUE"
        value = (
            operand[0] if operand_type == "Column Value" and column_op in ["is", "is one of"]
            else operand if column_op in ["is", "is NOT"]
            else operand[0]
        )

        return {
            column_name: {
                operator_map[column_op]: {
                    key: value
                }
            }
        }

    logical_op = raw_condition["Operator"]
    result = {logical_op: []}

    for cond in raw_condition["Conditions"]:
        if "Operator" in cond:  # Nested condition group
            nested_result = transform_condition(cond)
            result[logical_op].append(nested_result)
        else:  # Single condition
            column_name = cond["Column_Name"]
            column_op = cond["Column_Operator"]
            operand = cond["Operand"]
            operand_type = cond["Operand_Type"]
            mapped_op = operator_map.get(column_op)

            # Handle IS_EMPTY and IS_NOT_EMPTY with no operand
            if mapped_op in ["IS_EMPTY", "IS_NOT_EMPTY", "IS_MAXVAL", "IS_NOT_MAXVAL", "IS_MINVAL", "IS_NOT_MINVAL"]:
                transformed_cond = {
                    column_name: {
                        mapped_op: True
                    }
                }
            else:
                if not operand:
                    raise ValueError(f"Operand cannot be empty for condition on {column_name}")
                key = "COLUMN" if operand_type == "Column Value" else "VALUE"
                value = (operand[0] if operand_type == "Column Value" and column_op in ["is", "is one of"]
                         else operand if column_op in ["is one of", "is", "is NOT", "contains", "does NOT contain"]
                         else operand[0])
                transformed_cond = {
                    column_name: {
                        mapped_op: {
                            key: value
                        }
                    }
                }

            result[logical_op].append(transformed_cond)
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
        st.markdown(
            f"<div style='margin-left:{indent}px; padding: 5px 0;'>"
            f"<b>{data['Column_Name']}</b> "
            f"<span style='color: gray;'>({data['Column_Type']})</span> "
            f"'{data['Column_Operator']}' "
            f"<i>{', '.join(map(str, data['Operand']))}</i> "
            f"<span style='color: gray;'>({data['Operand_Type']})</span>"
            f"</div>",
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
