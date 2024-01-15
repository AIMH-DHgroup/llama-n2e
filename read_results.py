import os
import re
from difflib import SequenceMatcher
import json
import statistics
import nltk
import pandas as pd
#nltk.download('punkt')  # Download the sentence tokenizer model

if __name__ == '__main__':

    def clean_output(str1, str2):
        lines = str1.split('\n')
        phrases = nltk.sent_tokenize(str2)
        cleaned_text = []
        for line in lines:
            matcher = SequenceMatcher(None, line, str2, autojunk=False)
            match_ratio = matcher.ratio()
            if len(lines) > 20: # the more the text is fragmented into paragraphs, the lower the match with the individual paragraphs
                threshold = 0.06
            else:
                threshold = 0.09
            if round(match_ratio, 2) > threshold:
                cleaned_text.append(line)
            elif line != "":
                for phrase in phrases:
                    matcher = SequenceMatcher(None, line, phrase, autojunk=False)
                    match_ratio = matcher.ratio()
                    if match_ratio > 0.6:
                        cleaned_text.append(line)
            pattern = r'^Paragraph \d+: '  # Pattern for search
            if re.match(pattern, line):  # Check if the line matches the pattern
                cleaned_line = re.sub(pattern, '', line)
                cleaned_text.append(cleaned_line)
        str1 = "\n".join(cleaned_text)
        return str1

    def remove_tokens_to_match(str1, str2):
        str1 = clean_output(str1, str2)
        str1 = re.sub(r'\n{2,}', '\n', str1)
        return str1


    def print_lines_from_json(json_string, str2):
        json_data = json.loads(json_string)  # Decode the JSON object
        lines = json_data['paragraphs']  # Get the list of lines from the JSON object
        str1 = '\n'.join(lines)
        similarity = jaccard_similarity(str1, str2)
        print(f"The Jaccard similarity index between JSON file and the original text is: {similarity:.2f}\n")
        print_differences(str1, str2)
        print("\n")
        cleaned_text = []
        for line in lines:
            print(line)
            cleaned_text.append(line)
        cleaned_text = ' '.join(cleaned_text)
        return [similarity, cleaned_text]

    def create_json_from_text(text):
        lines = text.split('\n')  # Split text by newline character
        json_data = {
            'paragraphs': lines}  # Create JSON object with a key 'text_lines' containing the list of lines
        return json.dumps(json_data, indent=4)  # Convert JSON object to a formatted string

    def create_json_file(json_var, filename):
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(json_var, file, indent=4)
        if os.path.exists(filename):
            print(filename + " saved successfully.\n")
        else:
            print("Failed to save " + filename + "\n")

    def load_dict_from_file(file_name):
        with open(file_name, 'r', encoding='utf-8') as file:
            loaded_dict = json.load(file)
        return loaded_dict

    def jaccard_similarity(str1, str2):
        set1 = set(str1.split())
        set2 = set(str2.split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0


    def count_sentences(text):
        sentences = nltk.sent_tokenize(text)
        return len(sentences)

    def print_rows_and_paragraphs(str1):
        lines = str1.split('\n')
        paragraphs = 0
        sentences = []
        for line in lines:
            sentences.append(count_sentences(line))
            paragraphs += 1
        mean_sentences = 0
        for elem in sentences:
            index = sentences.index(elem)
            mean_sentences += sentences[index]
        mean_sentences /= len(sentences)
        len_chars = len(str1)
        tokens = nltk.word_tokenize(str1)
        num_tokens = len(tokens)
        return [paragraphs, sentences, mean_sentences, len_chars, num_tokens]

    def print_differences(str1, str2):
        set1 = set(str1.split())
        set2 = set(str2.split())

        similarity = jaccard_similarity(str1, str2)
        print(f"The Jaccard similarity index between the two text is: {similarity:.2f}\n")

        print(f"Differences between the two text:")
        print("----------")
        diff_1 = set1.difference(set2)
        diff_2 = set2.difference(set1)

        if len(diff_1) > 0:
            print("\'output text\' contains the following words not present in original text:")
            print(', '.join(diff_1))
        else:
            print("All words in \'output text\' are also present in original text")

        print("----------")

        if len(diff_2) > 0:
            print("Original text contains the following words not present in \'output text\'")
            print(', '.join(diff_2))
        else:
            print("All words in original text are also present in \'output text\'")


    def j_sim(str1, str2):
        similarity = jaccard_similarity(str1, str2)
        return similarity

    def analysis(input_dict, input_final):
        global df
        print("Total text outputs \"" + narrative_to_analyse + "\" with " + str(iterations) + " iterations: " + str(
            len(input_dict)))

        # prints all the outputs of the single narrative
        for key in input_dict:
            print("Output # " + str(key), "\n")
            print(input_dict[key], "\n\n")
            print("\nDifferences in normal paragraph division with original text:\n\n")
            print(print_differences(input_dict[key], narratives[narrative_titles.index(narrative_to_analyse)]),
                  "\n")

        # prints the differences between the outputs of the different models

        if len(narratives_dict) > 1:
            other_model_comparison = input("Enter second model to compare (full name): ")
            dictionary_to_analyse_with_other_model = narratives_dict[other_model_comparison][narrative_to_analyse]

            for key in input_dict:
                for key2 in dictionary_to_analyse_with_other_model:
                    print(
                        "Difference between Output # " + str(key) + " (" + model_to_analyse + ") with output # " + str(
                            key2) + " (" + other_model_comparison + ")\n")
                    print(print_differences(input_dict[key], dictionary_to_analyse_with_other_model[key2]), "\n\n")
                print("\n\n")

        # Compares all the outputs of all the models with the respective original texts and orders them from most similar to least similar

        models_analysis = []
        titles_analysis = []
        numberOutput_analysis = []
        narrative_analysis = []
        similarity = []

        for model, dictionary_title_outputs in narratives_dict.items():
            for title, outputs in dictionary_title_outputs.items():
                for i in outputs:
                    sim = j_sim(outputs[i], narratives[narrative_titles.index(title)])
                    models_analysis.append(model)
                    titles_analysis.append(title)
                    numberOutput_analysis.append(i)
                    narrative_analysis.append(outputs[i])
                    similarity.append(sim)

        # Combining lists into a list of tuples
        #combined_data = list(
        #    zip(models_analysis, titles_analysis, numberOutput_analysis, narrative_analysis, similarity))

        # Sort the list of tuples based on the 'similarity' column (fifth column, index 4)
        #sorted_data = sorted(combined_data, key=lambda x: x[4], reverse=True)

        # Decompose the sorted list of tuples into the original lists
        #models_analysis, titles_analysis, numberOutput_analysis, narrative_analysis, similarity = zip(*sorted_data)

        # Print the sorted results
        #print("models_analysis:", models_analysis)
        #print("titles_analysis:", titles_analysis)
        #print("numberOutput_analysis:", numberOutput_analysis)
        # print("narrative_analysis:", narrative_analysis)
        #print("similarity:", similarity)

        max_jac = 0
        best_json = {}
        best_text = ''
        result_tok = remove_tokens_to_match(input_final, narratives[narrative_titles.index(narrative_to_analyse)])
        json_result = create_json_from_text(result_tok)
        print("Reading paragraphs from JSON result:\n")
        jac, cleaned_text = print_lines_from_json(json_result,
                                                  narratives[narrative_titles.index(narrative_to_analyse)])
        if "Cannot find a match by removing tokens." not in result_tok:
            paragraphs, sentences, mean_sentences, chars, len_tokens = print_rows_and_paragraphs(result_tok)
            print("\nParagraphs: ", paragraphs, "\nSentences: ", sentences, "\nAvg sentences per paragraph: ",
                  round(mean_sentences, 1),
                  "\n# Characters: ", chars, "\n# Tokens: ", len_tokens)
            df_line = pd.DataFrame(
                {"narrative": narrative_to_analyse, "Jaccard": round(jac, 2), "# Paragraphs": paragraphs,
                 "Avg sentences": round(mean_sentences, 2), "# Characters": chars, "# Tokens": len_tokens},
                index=[3])
        else:
            print("\nData on paragraphs and sentences not available.\n")
            df_line = pd.DataFrame({"narrative": narrative_to_analyse, "Jaccard": jac}, index=[3])
        if jac > max_jac:
            max_jac = jac
            best_json = json_result
            best_text = narrative_to_analyse
        df = pd.concat([df, df_line]).reset_index(drop=True)
        if best_prompt:
            print("\nThe chosen output has Jaccard", round(max_jac, 2), " and prompt: " + best_prompt[
                narrative_to_analyse] + "\nWARNING: prompt read from saved json file. May not match exact results.\n")
        else:
            print("\nThe chosen output has Jaccard", round(max_jac, 2))
        create_json_file(best_json, 'paragraphs-' + best_text + '.json')

    print("################")
    print("#START ANALYSIS#")
    print("################\n")

    # Set suffix
    suffix = input("Enter suffix of the JSON files to upload: ")
    suffix_csv = input("Enter suffix of the CSV file to save: ")
    narratives_dict_output_filename = "narratives_dict_output-" + suffix + ".json"
    times_dict_filename = "times_dict-" + suffix + ".json"
    metadata_filename = "metadata-" + suffix + ".json"
    final_dict_filename = "final_dict-" + suffix + ".json"
    best_prompt_filename = "best_prompt-" + suffix + ".json"

    # Loading files
    narratives_dict = load_dict_from_file(narratives_dict_output_filename)
    times_dict = load_dict_from_file(times_dict_filename)
    metadata = load_dict_from_file(metadata_filename)
    if os.path.exists(final_dict_filename):
        final_dict = load_dict_from_file(final_dict_filename)
    else:
        final_dict = ''
    if os.path.exists(best_prompt_filename):
        best_prompt = load_dict_from_file(best_prompt_filename)
    else:
        best_prompt = ''

    # Loading metadata
    iterations = metadata["iterations"]
    narratives = metadata["narratives"]
    narrative_titles = metadata["narrative_titles"]

    # Times
    mean_times = []
    std_means = []
    sum_all = 0
    print("Execution times per model: \n")
    for key in times_dict.keys():
        print(key, "--> ", times_dict[key], "\n")
        mean = 0
        sum = 0
        count = 0
        std_values = []
        for model in times_dict[key]:
            for time in times_dict[key][model]:
                print(time)
                sum += time
                count += 1
                std_values.append(time)
                sum_all += time
            mean = float(sum / count)

        mean_times.append(mean)
        if len(std_values) > 1:
            std_means.append(statistics.stdev(std_values))
        else:
            std_means = 0

    print("Avg and std dev for each model:\n")
    for x in range(len(mean_times)):
        mean_s = (mean_times[x]*60) % 60
        if std_means != 0:
            dev_s = (std_means[x]*60) % 60
            print("Mean: ", int(mean_times[x]), "minutes and ", int(mean_s), "seconds, with dev std: ",
                  int(std_means[x]), "minutes and ", int(dev_s), "seconds. Sum of total times:", round(sum_all, 2))
        else:
            dev_s = 0
            print("Mean: ", int(mean_times[x]), "minutes and ", int(mean_s), "seconds. Sum of total times:", round(sum_all, 2))

    # Variables for dataframe construction
    df = pd.DataFrame({"narrative": [], "Jaccard": [], "# Paragraphs": [], "Avg sentences": [], "# Characters": [], "# Tokens": []})

    while True:
        print("\nOutputs available on the following models: ", list(narratives_dict.keys()))
        available_narratives = []

        for model in narratives_dict.keys():
            for narrative in narratives_dict[model].keys():
                if narrative not in available_narratives:
                    available_narratives.append(narrative)

        print("Outputs available on the following narratives: ", available_narratives)

        break_while = False
        break_while2 = False

        # Define the parameters for the analysis
        if len(narratives_dict.keys()) > 1:
            model_to_analyse = input("Insert a model to analyze: ")
            if model_to_analyse == 'exit':
                df_name = "results-" + suffix_csv + ".csv"
                df.to_csv(df_name, index=False)
                if os.path.exists(df_name):
                    print("\n", df_name, "saved successfully.\n")
                else:
                    print("\n", df_name, "not saved. Operation failed.\n")
                break
            program_break = False
            while model_to_analyse not in list(narratives_dict.keys()):
                model_to_analyse = input("Inserted model not present in the list.\nInsert a model to analyze: ")
                if model_to_analyse == 'exit':
                    program_break = True
                    break
            if program_break:
                df_name = "results-" + suffix_csv + ".csv"
                df.to_csv(df_name, index=False)
                if os.path.exists(df_name):
                    print("\n", df_name, "saved successfully.\n")
                else:
                    print("\n", df_name, "not saved. Operation failed.\n")
                break
        else:
            model_to_analyse = list(narratives_dict)[0]
            break_while = True

        if len(narratives_dict[model_to_analyse].keys()) > 1:
            narrative_to_analyse = input("Insert a narrative to analyze: ")
            if narrative_to_analyse == 'exit':
                df_name = "results-" + suffix_csv + ".csv"
                df.to_csv(df_name, index=False)
                if os.path.exists(df_name):
                    print("\n", df_name, "saved successfully.\n")
                else:
                    print("\n", df_name, "not saved. Operation failed.\n")
                break
            program_break = False
            while narrative_to_analyse not in available_narratives:
                if narrative_to_analyse == 'all':
                    break
                narrative_to_analyse = input("Inserted narrative not present in the list.\nInsert a narrative to analyse: ")
                if narrative_to_analyse == 'exit':
                    program_break = True
                    break
            if program_break:
                df_name = "results-" + suffix_csv + ".csv"
                df.to_csv(df_name, index=False)
                if os.path.exists(df_name):
                    print("\n", df_name, "saved successfully.\n")
                else:
                    print("\n", df_name, "not saved. Operation failed.\n")
                break
        else:
            narrative_to_analyse = list(narratives_dict[model_to_analyse])[0]
            break_while2 = True

        if narrative_to_analyse == 'all':
            for elem in available_narratives:
                narrative_to_analyse = elem
                dict_to_analyse = narratives_dict[model_to_analyse][narrative_to_analyse]
                final_to_analyse = final_dict[model_to_analyse][narrative_to_analyse]
                analysis(dict_to_analyse, final_to_analyse)
        else:
            dict_to_analyse = narratives_dict[model_to_analyse][narrative_to_analyse]
            final_to_analyse = final_dict[model_to_analyse][narrative_to_analyse]
            analysis(dict_to_analyse, final_to_analyse)

        if break_while and break_while2:
            df_name = "results-" + suffix + ".csv"
            df.to_csv(df_name, index=False)
            if os.path.exists(df_name):
                print("\n", df_name, "saved successfully.\n")
            else:
                print("\n", df_name, "not saved. Operation failed.\n")
            break