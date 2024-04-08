import os
import re
from difflib import SequenceMatcher
import json
import statistics
import nltk
import pandas as pd

if __name__ == '__main__':

    def clean_output(str1, str2):
        lines = str1.split('\n')
        phrases = nltk.sent_tokenize(str2)
        cleaned_text = []
        for line in lines:
            matcher = SequenceMatcher(None, line, str2, autojunk=False)
            match_ratio = matcher.ratio()
            if len(lines) > 20:  # the more the text is fragmented into paragraphs, the lower the match with the
                # individual paragraphs
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
        return similarity


    def create_json_from_text(text):
        lines = text.split('\n')
        json_data = {
            'paragraphs': lines}
        return json.dumps(json_data, indent=4)


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
        sent = nltk.sent_tokenize(text)
        return len(sent)


    def print_rows_and_paragraphs(str1, str2):
        str1 = re.sub(r'\n{2,}', '\n', str1)
        lines = str1.split('\n')
        phrases = nltk.sent_tokenize(str2)
        paragraphs = 0
        sent_list = []
        cleaned_text = []
        for line in lines:
            matcher = SequenceMatcher(None, line, str2, autojunk=False)
            match_ratio = matcher.ratio()
            if len(lines) > 20:  # the more the text is fragmented into paragraphs, the lower the match with the
                # individual paragraphs
                threshold = 0.06
            else:
                threshold = 0.09
            if round(match_ratio, 2) > threshold:
                if line.startswith(tuple(str(i) for i in range(10))) and line[len(str(int(line[0])))] == '.':
                    line = line[2:]
                    cleaned_text.append(line)
                else:
                    cleaned_text.append(line)
            elif line != "":
                for phrase in phrases:
                    matcher = SequenceMatcher(None, line, phrase, autojunk=False)
                    match_ratio = matcher.ratio()
                    if match_ratio > 0.6:
                        if line.startswith(tuple(str(i) for i in range(10))) and line[len(str(int(line[0])))] == '.':
                            line = line[2:]
                            cleaned_text.append(line)
                        else:
                            cleaned_text.append(line)
        text_cleaned = "\n".join(cleaned_text)
        lines = text_cleaned.split('\n')
        for line in lines:
            if line != "":
                sent_list.append(count_sentences(line))
                paragraphs += 1
        mean_sentences = 0
        for sent in sent_list:
            index = sent_list.index(sent)
            mean_sentences += sent_list[index]
        if len(sent_list) == 0:
            mean_sentences = 0
        else:
            mean_sentences /= len(sent_list)
        len_chars = len(text_cleaned)
        tok = nltk.word_tokenize(text_cleaned)
        num_tokens = len(tok)
        return [paragraphs, sent_list, mean_sentences, len_chars, num_tokens]


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

    def save_df(dataframe, is_narra):
        df_name = "results-" + suffix_csv + ".csv"
        new_row_names = list(narratives_dict.keys())
        dt = dataframe.rename(index=dict(zip(dataframe.index, new_row_names)))
        dt.to_csv(df_name)
        if os.path.exists(df_name):
            print("\n", df_name, "saved successfully.\n")
        else:
            print("\n", df_name, "not saved. Operation failed.\n")
        if is_narra:
            df_narra_name = "results-narra-" + suffix_csv + ".csv"
            df_narra.to_csv(df_narra_name)
            if os.path.exists(df_narra_name):
                print("\n", df_narra_name, "saved successfully.\n")
            else:
                print("\n", df_narra_name, "not saved. Operation failed.\n")

    def build_df(is_narra):
        global jac_list, par_list, prompt_list, sentences_list, chars_list, tokens_list, df, df_narra

        # this is for all outputs calculation

        # sum_val = 0
        # for val in jac_list:
        #    sum_val += val
        # mean_jac = float(sum_val / len(jac_list))

        sum_val = 0
        for val in best_jac_list:
            sum_val += val
        mean_best_jac = float(sum_val / len(best_jac_list))
        sum_val = 0
        for val in best_par_list:
            sum_val += val
        mean_best_par = float(sum_val / len(best_par_list))

        # also this

        # sum_val = 0
        # for val in par_list:
        #    sum_val += val
        # mean_par = float(sum_val / len(par_list))

        sum_val = 0
        for val in sentences_list:
            sum_val += val
        mean_sentence = float(sum_val / len(sentences_list))
        sum_val = 0
        for val in chars_list:
            sum_val += val
        mean_char = float(sum_val / len(chars_list))
        sum_val = 0
        for val in tokens_list:
            sum_val += val
        mean_token = float(sum_val / len(tokens_list))
        sum_val = 0
        for val in out_list:
            sum_val += val
        mean_out = float(sum_val / len(out_list))
        df_line = pd.DataFrame(
            {"Avg Jaccard": mean_best_jac, "Avg paragraphs": mean_best_par,
             "Avg sentences per p": mean_sentence, "Avg characters": mean_char,
             "Avg tokens": mean_token, "Avg outputs": mean_out}, index=[3])
        prompt_list = []
        jac_list = []
        par_list = []
        sentences_list = []
        chars_list = []
        tokens_list = []
        df = pd.concat([df, df_line]).reset_index(drop=True)
        if is_narra:
            df_narra_line = pd.DataFrame(
                {"narrative": available_narratives, "Jaccard": jac_narra, "# Paragraphs": par_narra,
                 "Avg sentences": sent_narra, "# Characters": char_narra,
                 "# Tokens": tokens_narra}
            )
            df_narra = pd.concat([df_narra, df_narra_line]).reset_index(drop=True)

    def analysis(input_dict):
        global df
        print("Total text outputs \"" + narrative_to_analyse + "\" with " + str(iterations) + " iterations: " + str(
            len(input_dict)))

        # prints all the outputs of the single narrative
        for out in input_dict:
            print("Output # " + str(out), "\n")
            print(input_dict[out], "\n\n")
            print("\nDifferences in normal paragraph division with original text:\n\n")
            print(print_differences(input_dict[out], narratives[narrative_titles.index(narrative_to_analyse)]),
                  "\n")

            json_result = create_json_from_text(input_dict[out])
            print("Reading paragraphs from JSON result:\n")
            jac = print_lines_from_json(json_result, narratives[narrative_titles.index(narrative_to_analyse)])
            paragraphs, sentences_seq, mean_sentences, chars, len_tokens = print_rows_and_paragraphs(
                input_dict[out], narratives[narrative_titles.index(narrative_to_analyse)]
            )
            print("\nParagraphs: ", paragraphs, "\nSentences: ", sentences_seq, "\nAvg sentences per paragraph: ",
                  round(mean_sentences, 1),
                  "\n# Characters: ", chars, "\n# Tokens: ", len_tokens)

            jac_list.append(jac)
            par_list.append(paragraphs)
            sentences_list.append(mean_sentences)
            chars_list.append(chars)
            tokens_list.append(len_tokens)
        out_len = len(input_dict.keys())
        out_list.append(out_len)


    def analysis_best(input_dict):
        global df
        print("Best output of ", narrative_to_analyse, "\n")
        print(input_dict, "\n\n")
        result_tok = remove_tokens_to_match(input_dict, narratives[narrative_titles.index(narrative_to_analyse)])
        print("\nDifferences in normal paragraph division with original text:\n\n")
        print(print_differences(result_tok, narratives[narrative_titles.index(narrative_to_analyse)]),
              "\n")

        json_result = create_json_from_text(result_tok)
        print("Reading paragraphs from JSON result:\n")
        jac = print_lines_from_json(json_result, narratives[narrative_titles.index(narrative_to_analyse)])
        paragraphs, sentences_seq, mean_sentences, chars, len_tokens = print_rows_and_paragraphs(result_tok, narratives[
            narrative_titles.index(narrative_to_analyse)])
        print("\nParagraphs: ", paragraphs, "\nSentences: ", sentences_seq, "\nAvg sentences per paragraph: ",
              round(mean_sentences, 1),
              "\n# Characters: ", chars, "\n# Tokens: ", len_tokens)
        return [jac, paragraphs, mean_sentences, chars, len_tokens]


    print("################")
    print("#START ANALYSIS#")
    print("################\n")

    # Set suffixes
    suffix = input("Enter suffix of the JSON files to upload: ")
    suffix_csv = input("Enter suffix of the CSV file to save: ")
    narra = input("Do you want also a complete narratives dataframe? (y/n): ")
    narratives_dict_output_filename = "narratives_dict_output-" + suffix + ".json"
    times_dict_filename = "times_dict-" + suffix + ".json"
    metadata_filename = "metadata-" + suffix + ".json"
    final_dict_filename = "final_dict-" + suffix + ".json"
    best_prompt_filename = "best_prompt-" + suffix + ".json"

    # Transform input 'narra' from string to boolean
    while narra.lower() != 'y' and narra.lower() != 'n':
        narra = input("Please insert a correct value (y/n): ")
    if narra.lower() == 'y':
        narra = True
    elif narra.lower() == 'n':
        narra = False

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
        sum_times = 0
        count = 0
        std_values = []
        for model in times_dict[key]:
            for time in times_dict[key][model]:
                print(times_dict[key][model][time])
                sum_times += times_dict[key][model][time]
                count += 1
                std_values.append(times_dict[key][model][time])
                sum_all += times_dict[key][model][time]
            mean = float(sum_times / count)

        mean_times.append(mean)
        if len(std_values) > 1:
            std_means.append(statistics.stdev(std_values))
        else:
            std_means = 0

    print("Avg and std dev for each model:\n")
    for x in range(len(mean_times)):
        mean_s = (mean_times[x] * 60) % 60
        if std_means != 0:
            dev_s = (std_means[x] * 60) % 60
            print("Mean: ", int(mean_times[x]), "minutes and ", int(mean_s), "seconds, with dev std: ",
                  int(std_means[x]), "minutes and ", int(dev_s), "seconds. Sum of total times:", round(sum_all, 2))
        else:
            dev_s = 0
            print("Mean: ", int(mean_times[x]), "minutes and ", int(mean_s), "seconds. Sum of total times:",
                  round(sum_all, 2))

    # Variables for dataframe construction
    jac_list = []
    par_list = []
    sentences_list = []
    chars_list = []
    tokens_list = []
    prompt_list = []
    out_list = []
    best_jac_list = []
    best_par_list = []
    jac_narra = []
    par_narra = []
    sent_narra = []
    tokens_narra = []
    char_narra = []
    df = pd.DataFrame(
        {"Avg Jaccard": [], "Avg paragraphs": [], "Avg sentences per p": [], "Avg characters": [],
         "Avg tokens": [], "Avg outputs": []})
    df_narra = pd.DataFrame(
        {"narrative": [], "Jaccard": [], "# Paragraphs": [], "Avg sentences": [], "# Characters": [], "# Tokens": []}
    )

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
                save_df(df, narra)
                break
            program_break = False
            while model_to_analyse not in list(narratives_dict.keys()):
                model_to_analyse = input("Inserted model not present in the list.\nInsert a model to analyze: ")
                if model_to_analyse == 'exit':
                    save_df(df, narra)
                    program_break = True
                    break
            if program_break:
                save_df(df, narra)
                break
        else:
            model_to_analyse = list(narratives_dict)[0]
            break_while = True

        if len(narratives_dict[model_to_analyse].keys()) > 1:
            narrative_to_analyse = input("Insert a narrative to analyze: ")
            if narrative_to_analyse == 'exit':
                save_df(df, narra)
                break
            program_break = False
            while narrative_to_analyse not in available_narratives:
                if narrative_to_analyse == 'all':
                    break
                narrative_to_analyse = input(
                    "Inserted narrative not present in the list.\nInsert a narrative to analyse: ")
                if narrative_to_analyse == 'exit':
                    build_df(narra)
                    program_break = True
                    break
            if program_break:
                save_df(df, narra)
                break
        else:
            narrative_to_analyse = list(narratives_dict[model_to_analyse])[0]
            break_while2 = True

        if narrative_to_analyse == 'all':
            for elem in available_narratives:
                max_jac = float(0)
                narrative_to_analyse = elem
                jaccard = float(0)
                n_par = 0
                sentences = float(0)
                characters = 0
                tokens = 0
                for prompt in narratives_dict[model_to_analyse][narrative_to_analyse]:
                    dict_to_analyse = narratives_dict[model_to_analyse][narrative_to_analyse]
                    analysis(dict_to_analyse[prompt])
                    dict_to_analyse = final_dict[model_to_analyse][narrative_to_analyse]
                    j, p, s, c, t = analysis_best(dict_to_analyse)
                    if j > max_jac:
                        jaccard = j
                        max_jac = j
                        n_par = p
                        sentences = s
                        characters = c
                        tokens = t
                jac_narra.append(jaccard)
                par_narra.append(n_par)
                sent_narra.append(sentences)
                char_narra.append(characters)
                tokens_narra.append(tokens)
                best_jac_list.append(jaccard)
                best_par_list.append(n_par)
                print("The best prompt is: ", best_prompt[model_to_analyse][narrative_to_analyse])
            build_df(narra)

        else:
            for prompt in narratives_dict[model_to_analyse][narrative_to_analyse]:
                dict_to_analyse = narratives_dict[model_to_analyse][narrative_to_analyse]
                analysis(dict_to_analyse)
                dict_to_analyse = final_dict[model_to_analyse][narrative_to_analyse]
                analysis_best(dict_to_analyse)
            print("The best prompt is: ", best_prompt[model_to_analyse][narrative_to_analyse])

        if break_while and break_while2:
            save_df(df, narra)
            break
