import csv
import json


def process_narrative(narrative):
    # Remove new line characters and wrap the narrative with [INST][/INST]
    cleaned_narrative = f"<s>[INST]{narrative.replace("\n","")}[/INST] "
    original_narrative = f"{narrative}</s>"
    return cleaned_narrative, original_narrative


def csv_to_json(input_csv, output_json):
    data = []

    with open(input_csv, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)

        # Assuming the first row contains headers, adjust if necessary
        headers = next(csv_reader)

        for row in csv_reader:
            narrative_column_index = headers.index('Narrazioni_in_paragrafi_WP')  # Update 'narrative' to the correct column header

            narrative = row[narrative_column_index]
            cleaned_narrative, original_narrative = process_narrative(narrative)

            data.append({
                'cleaned_narrative': cleaned_narrative,
                'original_narrative': original_narrative
            })

    with open(output_json, 'w') as json_file:
        json.dump(data, json_file, indent=2)

def json_to_list(input_json):
    result_list = []

    with open(input_json, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

        for item in data:
            combined_narratives = item['cleaned_narrative'] + item['original_narrative']
            result_list.append(combined_narratives)

    return result_list

def save_list_to_txt(output_txt, result_list):
    with open(output_txt, 'w', encoding='utf-8') as txt_file:
        for combined_narratives in result_list:
            txt_file.write(f"{combined_narratives}\n")


if __name__ == '__main__':

        input_csv_file = 'narra.csv'  # Replace with the actual input CSV file
        output_json_file = 'format_ft.json'  # Replace with the desired output JSON file

        csv_to_json(input_csv_file, output_json_file)
        input_json_file = 'format_ft.json'  # Replace with the actual output JSON file
        output_txt_file = 'format_ft.txt'  # Replace with the desired output text file

        result_list = json_to_list(input_json_file)
        save_list_to_txt(output_txt_file, result_list)