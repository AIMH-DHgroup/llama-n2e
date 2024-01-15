import csv
import json
import pandas as pd
import nltk
import os
#nltk.download('punkt')  # Download the sentence tokenizer model

if __name__ == '__main__':

    def jaccard_similarity(str1, str2):
        set1 = set(str1.split())
        set2 = set(str2.split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def print_differences(str1, str2):
        set1 = set(str1.split())
        set2 = set(str2.split())

        similarity = jaccard_similarity(str1, str2)
        print(f"The Jaccard similarity index between the two texts is: {similarity:.2f}\n")

        print(f"Differences between the two text:")
        print("----------")
        diff_1 = set1.difference(set2)
        diff_2 = set2.difference(set1)

        if len(diff_1) > 0:
            print("Text1 contains the following words not present in text2:")
            print(', '.join(diff_1))
        else:
            print("All words in text1 are also present in text2")

        print("----------")

        if len(diff_2) > 0:
            print("Text2 contains the following words not present in text1")
            print(', '.join(diff_2))
        else:
            print("All words in text2 are also present in text1")
        return similarity

    def count_sentences(text):
        sentences = nltk.sent_tokenize(text)
        return len(sentences)

    def print_rows_and_paragraphs(str1):
        # str1 = '\n'.join(str1)
        # lines = str1.split('\n')
        sentences = count_sentences(str1)
        len_chars = len(str1)
        tokens = nltk.word_tokenize(str1)
        num_tokens = len(tokens)
        return [sentences, len_chars, num_tokens]

    def print_lines_from_json(json_string1, json_string2, title):
        global df
        json_data1 = json.loads(json_string1)
        lines1 = json_data1['paragraphs']
        #str1 = '\n'.join(lines1)
        json_data2 = json.loads(json_string2)
        lines2 = json_data2['paragraphs']
        #str2 = '\n'.join(lines2)
        count = 1
        jac = 0
        print("\nStart", title, "narrative\n")
        if len(lines1) == len(lines2):
            for line in lines1:
                if not line:
                    pass
                else:
                    index = lines1.index(line)
                    print("\nDifferences in Paragraph", count, "are:\n")
                    jac = print_differences(line, lines2[index])
                    print("\nEnd differences in Paragraph", count, "\n")
                    count += 1
                    sentences, chars, len_tokens = print_rows_and_paragraphs(line)
                    df_line = pd.DataFrame(
                        {"narrative": title, "Jaccard": round(jac, 2), "# Sentences": round(sentences, 2),
                         "# Characters": chars, "# Tokens": len_tokens, "Text": line},
                        index=[3])
                    df = pd.concat([df, df_line]).reset_index(drop=True)
        else:
            if len(lines1) < len(lines2):
                for line in lines1:
                    if not line:
                        pass
                    else:
                        index = lines1.index(line)
                        print("\nDifferences in Paragraph", count, "are:\n")
                        jac = print_differences(line, lines2[index])
                        print("\nEnd differences in Paragraph", count, "\n")
                        count += 1
                        sentences, chars, len_tokens = print_rows_and_paragraphs(line)
                        df_line = pd.DataFrame(
                            {"narrative": title, "Jaccard": round(jac, 2), "# Sentences": round(sentences, 2),
                             "# Characters": chars, "# Tokens": len_tokens, "Text": line},
                            index=[3])
                        df = pd.concat([df, df_line]).reset_index(drop=True)
            else:
                for line in lines2:
                    if not line:
                        pass
                    else:
                        index = lines2.index(line)
                        print("\nDifferences in Paragraph", count, "are:\n")
                        jac = print_differences(line, lines1[index])
                        print("\nEnd differences in Paragraph", count, "\n")
                        count += 1
                        sentences, chars, len_tokens = print_rows_and_paragraphs(line)
                        df_line = pd.DataFrame(
                            {"narrative": title, "Jaccard": round(jac, 2), "# Sentences": round(sentences, 2),
                             "# Characters": chars, "# Tokens": len_tokens, "Text": line},
                            index=[3])
                        df = pd.concat([df, df_line]).reset_index(drop=True)
            print("\nThe number of paragraphs is different. Length of text1: ", len(lines1), ", length of text2:", len(lines2), "\n")
        print("\nEnd", title, "narrative\n")

    def create_json_from_text(text):
        lines = text.split('\n')
        json_data = {
            'paragraphs': lines}
        return json.dumps(json_data, indent=4)

    def load_dict_from_file(file_name):
        with open(file_name, 'r', encoding='utf-8') as file:
            loaded_dict = json.load(file)
        return loaded_dict

    with open('narra.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        columns = next(reader)

        index_column_p_wp = columns.index('Narrazioni_in_paragrafi_WP')
        index_column_p_ema = columns.index('Narrazione_divisa_in_Eventi_Ema')
        index_column_title = columns.index('Titolo')

        narratives_list_Wp = []
        narratives_list_Ema = []

        narratives_texts_list = []

        titles_list = []

        for line in reader:
            text_paragraphs_WP = line[index_column_p_wp]
            text_paragraphs_Ema = line[index_column_p_ema]

            title = line[index_column_title]

            paragraphs_WP = text_paragraphs_WP.split('\n\n')
            paragraphs_Ema = text_paragraphs_Ema.split('\n\n')

            if 'TODO' not in paragraphs_Ema:
                narratives_list_Ema.append(paragraphs_Ema)
            if 'TODO' not in paragraphs_WP:
                narratives_list_Wp.append(paragraphs_WP)

            titles_list.append(title)

            modified_text = text_paragraphs_WP.replace('\n\n', ' ')
            narratives_texts_list.append(modified_text)

    final_dict_filename = "final_dict-all-J99.json"

    ema_narratives = {}
    wp_narratives = {}
    final_dict = load_dict_from_file(final_dict_filename)
    df = pd.DataFrame({"narrative": [], "Jaccard": [], "# Sentences": [], "# Characters": [], "# Tokens": []})

    #print("\nEma narratives list:\n")

    for elem in narratives_list_Ema:
        index = narratives_list_Ema.index(elem)
    #    for p in elem:
    #        print(p, "\n")
        ema_narratives[titles_list[index]] = '\n'.join(elem)
    #    print("\n\n")

    #print("\nWP narratives list:\n")

    for elem in narratives_list_Wp:
        index = narratives_list_Wp.index(elem)
    #    for p in elem:
    #        print(p, "\n")
        wp_narratives[titles_list[index]] = '\n'.join(elem)
    #    print("\n\n")

    for key in ema_narratives.keys():
        ema_json = create_json_from_text(ema_narratives[key])
        final_json = create_json_from_text(final_dict["llama2:7b-chat-q8_0"][key])
        if key in wp_narratives.keys():
            wp_json = create_json_from_text(wp_narratives[key])
        else:
            wp_json = {}
            print("\nWARNING: No", key, "text in wp_narratives.\n")
        print_lines_from_json(ema_json, final_json, key) # the second argument must be the one produced by llama
    suffix = input("Insert suffix of the name of the dataset to save: ")
    df_name = "extract_narra-" + suffix + ".csv"
    df.to_csv(df_name, index=False)
    if os.path.exists(df_name):
        print("\n", df_name, "saved successfully.\n")
    else:
        print("\n", df_name, "not saved. Operation failed.\n")