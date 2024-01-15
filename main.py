from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)

import json
import time

import re
from difflib import SequenceMatcher
import nltk


# nltk.download('punkt')  # Download the sentence tokenizer model

if __name__ == '__main__':



    # FUNZIONI AUSILIARIE
    def save_dict_to_file(dictionary, file_name):
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(dictionary, file, ensure_ascii=False, indent=4)

    def clean_output(str1, str2):
        lines = str1.split('\n')
        phrases = str2.split('.')
        for phrase in phrases:
            index = phrases.index(phrase)
            phrases[index] += "."
        cleaned_text = []
        for line in lines:
            matcher = SequenceMatcher(None, line, str2, autojunk=False)
            match_ratio = matcher.ratio()
            if len(lines) > 20: # piu' il testo e' frammentato in paragrafi e piu' il match con i singoli paragrafi e' basso
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
        for line in lines:
            print(line)


    def create_json_from_text(text, filename):
        lines = text.split('\n')  # Split text by newline character
        json_data = {
            'paragraphs': lines}  # Create JSON object with a key 'text_lines' containing the list of lines
        return json.dumps(json_data, indent=4)  # Convert JSON object to a formatted string

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
        return [paragraphs, sentences, mean_sentences]

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

    models = ["llama2",
              "llama2:7b-chat",
              "llama2:7b-chat-q5_0",
              "llama2:7b-chat-q5_K_M",
              "llama2:7b-chat-q6_K",
              "llama2:7b-chat-q8_0",
              "llama2:7b-chat-fp16",
              "llama2:13b-chat-q3_K_M",
              "codellama:7b-code-q6_K",
              "codellama:7b",
              "codellama:7b-instruct"]

    mls_title = ["latest", "chat", "q5_0", "q5_K_M", "q6_K", "q8_0", "fp16", "13b-q3_K_M", "codellama-q6", "codellama", "codellama-instruct"]

    systemPrompts = [
        '''Write the narrative that I give you as input splitted into events writing the same exact sentences as they are written in the narrative without deleting, changing or adding any word.''',
        '''Divide in paragraphs the text provided, without adding, changing or deleting any words.''',
        '''Divide the provided text into paragraphs without deleting, adding or modifying or deleting any words, even those from different languages.''',
        '''### Instruction ###\nDivide the following text into paragraphs. Don't add or modify or delete any words.''',
        '''Divide the narrative I will give you into events. Then, for each event, attach the exact sentences to the event as they are written in the story. An event is a change of state, indicating the occurrence of something at a specific time and place and involving one or more participants.''',
        '''Split the narrative i will give you as input by chapters writing the same exact sentences as they are written in the narrative without deleting, changing or adding any word.''']
    syspts_title = ["write", "paragraph", "paragraph2", "paragraph3", "divide", "chapter"]
    narrazioni = [
        '''Many monsters appear in old sea stories. Among the others, the Aspidochelone (or Fastitocalon), a creature similar to an island; the bishop-fish (also known as the sea bishop, sea monk or monk-fish), a sea monster whose legend might also have originated from encounters with seals, sharks or walruses; and, even the biblical Leviathan. Myth around these creatures has lived at least up to mid 1800 and is still alive in some cultures. The Greek philosopher Aristotle (fourth century BC), in his work "The history of animals", distinguished the common squid (teuthis) from its larger and rarer cousin (teuthus or, in some translations, teuthos), which could reach 2.3m in length. Roman naturalist Pliny the Elder (first century AD), in his work "Natural history", described a giant squid, which had a body “as large as a barrel” and tentacles reaching 9.1m in length. Curiously, In Pliny's (as well as in Aristotle's) works, the giant squid was treated as a common animal and did not belong to myths of their ages. However, some later authors, especially in fictional literature, made this mistake. In fact, the Kraken’s legend originated from sailors’ accounts dating from a millennium after the age of Pliny. The legend of the Kraken was born from seamen’s stories, but it was much modified and strengthened over the years. Right from the start, the Kraken was universally incorporated into Nordic mythology and folklore. According to an obscure, ancient manuscript of circa 1180 by King Sverre of Norway, the Kraken was just one of many sea monsters. Still, it had its own peculiarities: it was colossal in size, as large as an island, and capable of sinking ships; it haunted the seas between Norway and Iceland, and between Iceland and Greenland.''',
        '''The Florentine humanist and chancellor Leonardo Bruni (Arezzo, 1370 - Florence, March 9, 1444) recounts the journey to Constance he made between November and December 1414 in a letter from his private epistolary (IV 3 according to the edition edited by Lorenzo Mehus, Leonardi Bruni Aretini Epistolarum libri VIII, I, Florentiae, ex typographia Bernardi Paperinii, 1741, pp. 102-109) addressed to his friend Niccolò Niccoli, also a humanist in the Florentine circle. Bruni, who in those years held the post of apostolic secretary on behalf of Pope John XXIII, headed to the German city along with the entourage of curials in the pontiff's service to attend the work of the Council, which met in Constance to reestablish the unity of the Catholic and Orthodox Churches and thus end the Western Schism. In order to reach Pope John XXIII, who made his entrance in Constance as early as October 28, Bruni had to move from Rome with the other members of the Curia a few weeks later, for in the letter he states that he arrived in Verona on November 27 («Veronam ad V Kalendas Decembris cum satis mane venissem») and stayed there for two days, during which he had the opportunity to admire some of the city’s principal beauties, «nonnulla antiqua et visu digna». Bruni’s transalpine journey to Constance begins from this city.''',
        '''Once upon a time, in the mystical land of Eldoria, where magic flowed like a gentle river and fantastical creatures roamed freely, there lived a young sorcerer named Aria. Aria possessed a rare gift - the ability to communicate with the ancient spirits that dwelled in the heart of the Enchanted Forest. Her emerald-green eyes sparkled with the wisdom of the ages, and her long, flowing hair shimmered like moonlight. Eldoria was a realm of balance, where light and darkness coexisted in harmony. However, an ominous shadow had begun to creep across the land, threatening to disrupt the delicate equilibrium. Aria sensed the growing unrest among the spirits and knew that she was destined to play a crucial role in restoring the balance. Guided by a vision, Aria set out on a quest to seek the Oracle of Whispers, a mystical being who resided atop the Silver Peak, the tallest mountain in Eldoria. Legends spoke of the Oracle's ability to unveil the secrets of the past, present, and future. Aria hoped that the Oracle would provide her with the knowledge needed to confront the encroaching darkness. As she ventured deeper into the Enchanted Forest, Aria encountered magical creatures who became her steadfast companions. There was Luna, a graceful unicorn with a silvery mane that glowed with ethereal light, and Ember, a mischievous fire sprite who danced around Aria, leaving trails of warmth and light in her wake.''',
        '''During the Battle of Britain in 1940, a squadron of Royal Air Force (RAF) pilots, stationed in England, engaged in relentless dogfights against the Luftwaffe, the German air force. Among these pilots was Flight Lieutenant James, a skilled and courageous flyer known for his daring maneuvers and quick thinking. One day, during an intense aerial combat, James found himself outnumbered by several enemy Messerschmitt fighters. Despite being in a dire situation, he remained calm and focused. With exceptional skill, he maneuvered his Spitfire through the chaotic sky, evading enemy fire and weaving through the clouds. As the dogfight ensued, James noticed one of his fellow pilots, Flight Officer Carter, in trouble. Carter's aircraft was heavily damaged and was losing altitude rapidly. Realizing the imminent danger, James immediately broke away from his engagement with the enemy and dove towards Carter's stricken plane. With precision flying, James positioned his Spitfire underneath Carter's failing aircraft. Braving the enemy fire, he signaled to Carter to bail out while he attempted to stabilize the damaged plane. Carter, albeit shaken, managed to eject safely moments before his aircraft plummeted to the ground and exploded. James, now alone and vastly outnumbered, skillfully fought off the pursuing German fighters, using every ounce of his flying expertise to outmaneuver and evade them. Despite the odds, he managed to evade the enemy long enough to make a daring escape back to the safety of his airbase. His selfless act of risking his own safety to save a fellow pilot and his exceptional aerial skills in the face of overwhelming odds became a legend among his squadron. Flight Lieutenant James' bravery and quick thinking not only saved a comrade but also showcased the unwavering courage displayed by many RAF pilots during the critical moments of World War II.''',
        '''As the sun was setting on August 18th 2003, the night fishermen of Hahaya village eased their wooden pirogues off the jagged lava rocks and slid into the water. The ocean off the western coast of Grande Comore was calm and as the half-moon rose, they could see the volcano of Karthala silhouetted against the darkening sky. A few hundred metres offshore, one of the fishermen, a veteran of decades of nights on the dark water, laid his paddles across the boat and prepared a line. He tied two flat black stones above a baited hook, then let the fine filament slip through his fingers until it touched the seabed, deep below. He was waiting for the nibble and tug of a fish—a snapper or a grouper, perhaps, or if he was lucky, a marlin, which he would take the next morning to sell at the market in Moroni. But this time the tug was unfamiliar, and the old fisherman fought with the line before he managed to pull the fish to the surface. Deep water at night is ink-black and the first thing he saw was a pair of eyes, glowing pink in the pale moonlight. As they surfaced, he could make out a large fish. He recognised it instantly as a gombessa, or coelacanth (pronounced see-la-kanth). Although rarely caught, it was known to all in the Comoros as their most precious asset, a fish that some said was the ancestor of man. Only six coelacanths had been caught in the waters off Hahaya since 1966, and none in the previous five years, but the old fisherman knew what to do. He tethered it to the back of the boat and paddled back to the village. He knew there was little time to lose as gombessa live in the ocean depths and had never survived for more than a few hours at the surface. Determined to try, he made a safe water pool, and waited for the sun to rise.''',
        '''In the distant galaxy of Tatooine, where twin suns cast long shadows across the vast desert, a young scavenger named Rey stumbled upon a mysterious artifact buried in the sands. As she dusted off the object, a holographic message flickered to life – it was a distress call from Leia Organa. Leia, leader of the Resistance, pleaded for Rey's help. The remnants of the First Order had regrouped under a new sinister leader, and a dark force threatened to plunge the galaxy into chaos once more. Rey, driven by a sense of duty and a mysterious connection to the Force, embarked on a journey to gather allies. Her path intertwined with Poe Dameron, the daring pilot, and Finn, a former stormtrooper seeking redemption. Together, they faced perilous challenges, navigating through treacherous worlds and battling the shadowy remnants of the First Order. Alongside them stood an unlikely ally – Kylo Ren, torn between the darkness and a lingering pull to the light. As the group ventured deeper into the galaxy, ancient secrets were unraveled. They discovered a forgotten Jedi temple on a distant moon, and Rey underwent rigorous training to harness her latent Force abilities. Meanwhile, whispers of a prophesized savior echoed through the stars, and the group realized that their destinies were intertwined with the very fabric of the Force. In a climactic showdown, Rey faced the new threat, revealing the strength she had cultivated through her journey. The galaxy stood at the precipice of a new era, and the balance of the Force hung in the balance. With the combined efforts of Rey, Finn, Poe, and even Kylo Ren, the dark forces were vanquished, and peace was restored. As the twin suns set over Tatooine, Rey looked to the horizon, knowing that the Force would continue to guide the destiny of the galaxy. The heroes had forged a new legacy, and the story of their triumph echoed through the stars, a beacon of hope for generations to come.''',
        '''Meet Emily Rodriguez, a trailblazing businesswoman whose indomitable spirit and innovative vision have left an indelible mark on the corporate landscape. Born on November 30, 1980, in New York City, Rodriguez exhibited an early passion for entrepreneurship and leadership. she shap her destiny with a determination that would later redefine the business landscape. After earning her MBA from a prestigious business school, Rodriguez embarked on her professional journey, initially working for a Fortune 500 company where she quickly rose through the ranks due to her exceptional strategic acumen and dedication. However, her true calling beckoned her to forge her path. In 2010, Rodriguez founded her own tech startup, a bold move that reflected her foresight in the burgeoning technology sector. Her company swiftly gained acclaim for groundbreaking innovations in artificial intelligence, revolutionizing how businesses approached data analytics. Rodriguez's leadership style, marked by a balance of empathy and decisiveness, fostered a culture of collaboration and creativity within her team. As the CEO, she navigated her company through challenges, adeptly steering it towards sustained growth. Rodriguez's commitment to diversity and inclusion also became evident as she championed initiatives to empower women in the tech industry. Her influence extended beyond the boardroom, with Rodriguez frequently sought after as a keynote speaker at conferences, where she shared insights on entrepreneurship, leadership, and the future of technology. Over the years, Rodriguez's achievements garnered recognition, earning her a spot on various influential business lists. Her philanthropic efforts further highlighted her commitment to social responsibility, as she directed resources towards education and community development. As Emily Rodriguez celebrates her 1-year anniversary, her journey stands as a testament to the power of resilience, innovation, and unwavering determination. A pioneer in the business world, she continues to inspire the next generation of entrepreneurs, proving that with passion and purpose, any obstacle can be transformed into an opportunity for success.''',
        '''In the heart of the Pacific Northwest, nestled between towering evergreen trees and rolling hills, lies the serene enclave of Cedar Valley. This picturesque natural haven is a sanctuary for those seeking solace amidst the beauty of untouched wilderness. It is situated at a latitude of 47.356° N and a longitude of 122.189° W. As the sun begins its daily descent, casting a warm golden glow upon the landscape, the temperature hovers around a comfortable 68°F (20°C). The valley, at an elevation of 500 feet (152 meters), enjoys a mild climate year-round, making it an idyllic retreat for nature enthusiasts. The heartbeat of Cedar Valley is the crystal-clear Cedar Creek, meandering through the landscape like a silken ribbon. This pristine watercourse originates from the snow-capped peaks of the Cascade Range, ensuring a constant flow that nurtures the rich biodiversity within the valley. The creek, with a width averaging 15 feet (4.5 meters), provides a habitat for various aquatic species, including rainbow trout and native salmon. Flora adorns the valley in a vibrant tapestry of colors. Douglas fir and western red cedar trees dominate the skyline, their towering heights reaching up to 200 feet (61 meters). The forest floor beneath is carpeted with a lush undergrowth of ferns, huckleberry bushes, and delicate trillium flowers. Wildlife thrives in this natural haven, with black-tailed deer grazing peacefully in the meadows, red-tailed hawks soaring overhead, and the occasional glimpse of a black bear lumbering through the dense foliage. The symphony of nature's sounds, from the babbling creek to the rustling leaves, creates a tranquil melody that resonates throughout the valley. Cedar Valley stands as a testament to the intrinsic beauty of the natural world, inviting visitors to immerse themselves in its quiet splendor and connect with the untamed essence of the Pacific Northwest.''',
        '''Once upon a time, on a quaint farm nestled between rolling hills and lush meadows, a diverse community of animals coexisted in harmony. The mornings were heralded by the cheerful crowing of the rooster, signaling the beginning of a new day. The industrious ants marched tirelessly, working together to build intricate tunnels beneath the soil, while the wise old owl watched over the farm, dispensing sagacious advice to any creature seeking it. Amidst the golden fields of wheat and the fragrant orchards, a mischievous group of goats frolicked with boundless energy. Their antics brought joy to the entire farm, whether they were skillfully navigating rocky outcrops or engaging in spirited games of tag. In the shade of ancient oak trees, the pigs lazily wallowed in the cool mud, creating a symphony of contented grunts and squeals that echoed through the serene landscape. The farm's most endearing residents were the ducks that gracefully glided across the tranquil pond, their feathers glistening in the sunlight. Each day, they orchestrated a synchronized swim, weaving intricate patterns on the water's surface. Meanwhile, the diligent honeybees buzzed around vibrant blossoms, diligently collecting nectar to produce the farm's golden honey, a sweet testament to their hard work. As the sun dipped below the horizon, casting a warm glow over the farm, the animals gathered for a communal evening ritual. The melodious chirping of crickets served as a backdrop to their shared moments of peace and camaraderie. Underneath the vast canvas of the night sky, the farm became a haven where the bonds between its diverse inhabitants flourished, creating a tapestry of life woven with the threads of friendship and cooperation. And so, the farm's story continued, a testament to the beauty that emerges when different creatures come together to create a harmonious tapestry of life.''',
        '''Nintendo Co., Ltd.[b] is a Japanese multinational video game company headquartered in Kyoto. It develops, publishes and releases both video games and video game consoles. Nintendo was founded in 1889 as Nintendo Koppai[c] by craftsman Fusajiro Yamauchi and originally produced handmade hanafuda playing cards. After venturing into various lines of business during the 1960s and acquiring a legal status as a public company, Nintendo distributed its first console, the Color TV-Game, in 1977. It gained international recognition with the release of Donkey Kong in 1981 and the Nintendo Entertainment System and Super Mario Bros. in 1985. Since then, Nintendo has produced some of the most successful consoles in the video game industry: the Game Boy, the Super Nintendo Entertainment System, the Nintendo DS, the Wii, and the Switch. It has created numerous major franchises, including Mario, Donkey Kong, The Legend of Zelda, Metroid, Fire Emblem, Kirby, Star Fox, Pokémon, Super Smash Bros., Animal Crossing, Xenoblade Chronicles, and Splatoon, and Nintendo's mascot, Mario, is internationally recognized. The company has sold more than 5.592 billion video games and over 836 million hardware units globally, as of March 2023. In early 2023, the Super Nintendo World theme park area in Universal Studios Hollywood opened. The Super Mario Bros Movie was released on 5 April 2023, and has grossed over $1.3 billion worldwide, setting box-office records for the biggest worldwide, opening weekend for an animated film, the highest-grossing film based on a video game and the 15th-highest-grossing film of all-time. Nintendo has multiple subsidiaries in Japan and abroad, in addition to business partners such as HAL Laboratory, Intelligent Systems, Game Freak, and The Pokémon Company. Nintendo and its staff have received awards including Emmy Awards for Technology & Engineering, Game Awards, Game Developers Choice Awards, and British Academy Games Awards. It is one of the wealthiest and most valuable companies in the Japanese market.''',
        '''In news that won’t come as a surprise to anyone who’s been feeling the pinch, the global cost of living crisis is far from over. And big-city dwellers can really take a hit. According to the annual Worldwide Cost of Living Index that’s published by the Economist Intelligence Unit (EIU), the average cost of living rose by 7.4% this year. Grocery prices increased the fastest. Although this is slightly lower than the 8.1% jump the same survey recorded in 2022, the numbers remain significantly higher than “historic trends.” But there is some good news. Utility prices, the fastest rising category in the 2022 survey, showed the least amount of inflation this time around. Price increases are slowing in pace because of the waning of supply chain issues since China lifted its Covid-19 restrictions in late 2022. However, grocery prices are continuing to rise as retailers pass on higher costs to consumers. “We expect inflation to continue to decelerate in 2024, as the lagged impact of interest-rate rises starts affecting economic activity, and in turn, consumer demand,” Upasana Dutt, Head of Worldwide Cost of Living at EIU, said in a statement. Dutt went on to warn the upside risks of armed conflict and extreme weather remain. “Further escalations of the Israel-Hamas war would drive up energy prices, while a greater than expected impact from El Niño would push up food prices even further,” she added. Inevitably, increasing living costs have meant that many cities have become more expensive to live in – but some get hit harder than others. The city-state of Singapore and Switzerland’s Zurich were named as the most expensive cities in the world. The rise of the latter, which jumped from sixth place on last year’s list, was attributed to the strength of the Swiss Franc along with the high prices of groceries, household goods and recreation. Singapore’s costly transport and clothing were also noted.''',
        '''Arctic sea ice likely reached its annual minimum extent on Sept. 19, 2023, making it the sixth-lowest year in the satellite record, according to researchers at NASA and the National Snow and Ice Data Center (NSIDC). Meanwhile, Antarctic sea ice reached its lowest maximum extent on record on Sept. 10 at a time when the ice cover should have been growing at a much faster pace during the darkest and coldest months. Scientists track the seasonal and annual fluctuations because sea ice shapes Earth’s polar ecosystems and plays a significant role in global climate. Researchers at NSIDC and NASA use satellites to measure sea ice as it melts and refreezes. They track sea ice extent, which is defined as the total area of the ocean in which the ice cover fraction is at least 15%. Between March and September 2023, the ice cover in the Arctic shrank from a peak area of 5.64 million square miles (14.62 million square kilometers) to 1.63 million square miles (4.23 million square kilometers). That’s roughly 770,000 square miles (1.99 million square kilometers) below the 1981–2010 average minimum of 2.4 million square miles (6.22 million square kilometers). The amount of sea ice lost was enough to cover the entire continental United States. Sea ice around Antarctica reached its lowest winter maximum extent on Sept. 10, 2023, at 6.5 million square miles (16.96 million square kilometers). That’s 398,000 square miles (1.03 million square kilometers) below the previous record-low reached in 1986 – a difference that equates to roughly the size of Texas and California combined. The average maximum extent between 1981 and 2010 was 7.22 million square miles (18.71 million square kilometers). “It’s a record-smashing sea ice low in the Antarctic,” said Walt Meier, a sea ice scientist at NSIDC. “Sea ice growth appears low around nearly the whole continent as opposed to any one region.”''',
        '''Ada Lovelace, born Augusta Ada Byron in 1815, was the only legitimate child of the renowned poet Lord Byron. Raised by her mother, Lady Anne Isabella Milbanke, Ada showed an early aptitude for mathematics. Her education in mathematics and science was guided by tutors such as Mary Somerville and Augustus De Morgan. Ada's remarkable analytical skills and curiosity soon led her to the work of Charles Babbage, a mathematician and inventor. In her annotated translation of an article about Babbage's Analytical Engine, Ada introduced notes that extended far beyond mere translation. These notes, which some consider to be the first computer program, detailed how the Analytical Engine could manipulate symbols beyond numbers, including music and art. Ada's visionary ideas laid the foundation for modern programming. "La máquina de Babbage no es simplemente una calculadora gigante, sino una herramienta capaz de crear arte y música mediante el manejo de símbolos abstractos", she wrote, emphasizing the potential artistic applications of Babbage's invention. Her insights extended beyond the technical realm, with Ada contemplating the broader societal implications of computing. In a letter to her friend Charles Wheatstone, she articulated, "Les machines de Babbage pourraient un jour influencer profondément la société, modifiant la façon dont nous pensons et travaillons." (Babbage's machines could one day profoundly influence society, altering how we think and work.) Ada Lovelace's tragically short life ended in 1852 at the age of 36. Despite her untimely death, her legacy endures, and her contributions to the field of computer science continue to inspire generations of scientists and programmers. "Ada Lovelace a ouvert la voie à une ère où les femmes jouent un rôle crucial dans la science et la technologie." (Ada Lovelace paved the way for an era where women play a crucial role in science and technology.)''',
        '''In a picturesque village nestled amid rolling hills, there lived a wise old scholar named Lucius. Lucius, known for his erudition, spent his days immersed in ancient texts, seeking wisdom from the past to illuminate the present. One day, as he strolled through the quaint cobblestone streets, he stumbled upon a mysterious tome in an antiquarian bookstore. The title, "Lux et Veritas," whispered promises of enlightenment and truth. Intrigued, Lucius opened the book to find a Latin inscription that read, "Sapientia est potentia," reminding him that knowledge is power. Eager to unravel the secrets within, he delved into the pages, each filled with arcane knowledge and cryptic symbols. As Lucius continued his exploration, he came across a passage that struck a chord deep within him: "Tempus fugit." The reminder that time flies resonated with him, urging him to make the most of every moment. Inspired, he decided to embark on a journey to share his newfound wisdom with the villagers. In the heart of the village square, Lucius gathered the townsfolk and began to impart his insights. With eloquence, he quoted Seneca, saying, "Non scholae sed vitae discimus," emphasizing that we learn not for school but for life. The villagers listened attentively as Lucius wove the wisdom of the ancients into the fabric of their everyday lives. As seasons changed and years passed, Lucius became a revered figure in the village, guiding generations with the timeless words of Cicero: "Salus populi suprema lex esto," let the welfare of the people be the supreme law. The village flourished under Lucius's tutelage, and his teachings echoed through the ages. On the eve of his 80th birthday, Lucius gathered the villagers one last time. With a twinkle in his eye, he shared a final Latin aphorism: "Vivere est vincere," to live is to conquer. Lucius, having conquered the limitations of time, bid farewell to the mortal realm, leaving behind a legacy steeped in Latin wisdom that would endure for centuries.''',
        '''In the heart of the 19th century, amidst the fervor of the California Gold Rush, a daring prospector named Samuel Thornton embarked on a treacherous journey that would etch his name into the annals of history. Armed with little more than a weathered map and an unyielding spirit, Thornton sets out from the bustling streets of San Francisco in search of rumored riches hidden deep within the Sierra Nevada mountains. His odyssey began with the arduous trek through dense forests, where towering redwoods whispered tales of the untamed wilderness. Samuel, clad in worn leather, pushed through the thicket, his every step a negotiation with nature itself. As he ascended, the air grew thinner, and the distant echo of rushing rivers beckoned him forward. Crossing the formidable Sierra Nevada range proved to be a formidable challenge. Battling blizzards and avalanches, Thornton's journey became a dance with death. Yet, undeterred, he pressed on, driven by the promise of fortune that glittered in the icy streams below. The next leg unfolded in the scorching Nevada desert, where the once icy winds were replaced by a relentless sun that beat down on him. Water sources were scarce, and Thornton's canteen became a lifeline in this unforgiving terrain. Bandits and nomadic tribes added an element of danger, forcing him to remain vigilant at every turn. Venturing through ghost towns and abandoned settlements, Thornton faced the haunting echoes of shattered dreams. The eerie silence spoke of ambitions crushed by the harsh realities of the West. Undeterred, he pressed on, the ghostly reminders propelling him forward. '''
    ]
    titoliNarrazione = ["calamaro", "bruni", "gpt-fantasy", "gpt-ww2", "celacanto", "starW", "buisnessWoman", "naturalPlace", "animalFarm", "nintendo", "expensiveCities", "ArcticSea", "AdaByron", "latin", "journeyPiuBreve"]

    ###########
    #input qui#
    ###########

    chosen_models = ["q8_0"]    # model(s)
    chosen_systemPrompt = ["paragraph"]   # system prompt(s)
    jaccard_prompt = ["paragraph2", "paragraph3"] # prompt to use in case of low jaccard
    iterazioniSuOgniNarrazione = 1  # iterations per text
    narrazioni_scelte = ["bruni"]   # text(s)
    jaccard_threshold = 0.9 # threshold that triggers prompts in jaccard_prompt

    ############
    #fine input#
    ############

    dizionario_narrazioni = {}

    dizionario_tempo = {}

    dizionario_finale = {}

    systemPrompt = []

    jaccard_systemPrompt = []

    modelToIterate = []

    best_prompt = {}

    for prompt in chosen_systemPrompt:
        if prompt in syspts_title:
            systemPrompt.append(str(systemPrompts[syspts_title.index(prompt)]))
        else:
            raise Exception("systemPrompt not in list.")

    for prompt in jaccard_prompt:
        if prompt in syspts_title:
            jaccard_systemPrompt.append((systemPrompts[syspts_title.index(prompt)]))
        else:
            raise Exception("Jaccard systemPrompt not in list.")

    for model in chosen_models:

        if model in mls_title:
            modelToIterate.append(models[mls_title.index(model)])
        else:
            raise Exception("model not in list.")

    for model in modelToIterate:

        for narrazione, titolo in zip(narrazioni, titoliNarrazione):

            if titolo in narrazioni_scelte:

                dizionario_stringhe = {}
                tempi = []
                events_res = ''
                max_jac = 0

                for prompt in systemPrompt:

                    for _ in range(iterazioniSuOgniNarrazione):

                        print("Inizio iterazione #" + str(_) + " sul testo \"" + titolo + "\" con modello " + model + "\nsystemPrompt: " + prompt)
                        print("\n")

                        json_schema = {
                           "paragraphs": []
                        }
                        example_output_json = json.dumps(json_schema, indent=2)
                        template = prompt + " Use the following JSON schema:\n\n" + example_output_json + "\n\n" + "Now, considering the schema, " + prompt + "\nText:\n" + narrazione

                        llm = Ollama(
                            model=model,
                            system=prompt,
                        #    format='json',
                        #    template=template,
                            num_ctx=4096,
                            temperature=0.01,
                            #repeat_last_n=0,   #<-- output piu' precisi ma spezzati in piu' paragrafi
                            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                        )

                        #chat_model = ChatOllama(
                        #    model=model,
                        #    format="json",
                        #    #system=prompt,
                        #    num_ctx=4096,
                        #    temperature=0.01,
                        #    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                        #)

                        #example_narration = '''Upon graduating in September 1785, Bonaparte was commissioned a second lieutenant in La Fère artillery regiment. He served in Valence and Auxonne until after the outbreak of the French Revolution in 1789. Bonaparte was a fervent Corsican nationalist during this period. He asked for leave to join his mentor Pasquale Paoli, when Paoli was allowed to return to Corsica by the National Assembly. But Paoli had no sympathy for Napoleon, as he deemed his father a traitor for having deserted the cause of Corsican independence. He spent the early years of the Revolution in Corsica, fighting in a complex three-way struggle among royalists, revolutionaries, and Corsican nationalists. Napoleon embraced the ideals of the Revolution, becoming a supporter of the Jacobins and joining the pro-French Corsican Republicans who opposed Paoli's policy and his aspirations to secede. He was given command over a battalion of volunteers and promoted to captain in the regular army in 1792, despite exceeding his leave of absence and leading a riot against French troops. When Corsica declared formal secession from France and requested the protection of the British government, Napoleon and his commitment to the French Revolution came into conflict with Paoli, who had decided to sabotage the Corsican contribution to the Expédition de Sardaigne by preventing a French assault on the Sardinian island La Maddalena. Bonaparte and his family were compelled to flee to Toulon on the French mainland in June 1793 because of the split with Paoli. Although he was born "Napoleone Buonaparte", it was after this that Napoleon began styling himself "Napoléon Bonaparte". His family did not drop the name Buonaparte until 1796. The first known record of him signing his name as Bonaparte was at the age of 27 (in 1796). In July 1793, Bonaparte published a pro-republican pamphlet, Le souper de Beaucaire (Supper at Beaucaire), which gained him the support of Augustin Robespierre, the younger brother of the Revolutionary leader Maximilien Robespierre.'''
                        #example_output = '''Upon graduating in September 1785, Bonaparte was commissioned a second lieutenant in La Fère artillery regiment. He served in Valence and Auxonne until after the outbreak of the French Revolution in 1789.\n\nBonaparte was a fervent Corsican nationalist during this period. He asked for leave to join his mentor Pasquale Paoli, when Paoli was allowed to return to Corsica by the National Assembly. But Paoli had no sympathy for Napoleon, as he deemed his father a traitor for having deserted the cause of Corsican independence. He spent the early years of the Revolution in Corsica, fighting in a complex three-way struggle among royalists, revolutionaries, and Corsican nationalists.\n\nNapoleon embraced the ideals of the Revolution, becoming a supporter of the Jacobins and joining the pro-French Corsican Republicans who opposed Paoli's policy and his aspirations to secede. He was given command over a battalion of volunteers and promoted to captain in the regular army in 1792, despite exceeding his leave of absence and leading a riot against French troops. When Corsica declared formal secession from France and requested the protection of the British government, Napoleon and his commitment to the French Revolution came into conflict with Paoli, who had decided to sabotage the Corsican contribution to the Expédition de Sardaigne by preventing a French assault on the Sardinian island La Maddalena.\n\nBonaparte and his family were compelled to flee to Toulon on the French mainland in June 1793 because of the split with Paoli. Although he was born "Napoleone Buonaparte", it was after this that Napoleon began styling himself "Napoléon Bonaparte". His family did not drop the name Buonaparte until 1796. The first known record of him signing his name as Bonaparte was at the age of 27 (in 1796).\n\nIn July 1793, Bonaparte published a pro-republican pamphlet, Le souper de Beaucaire (Supper at Beaucaire), which gained him the support of Augustin Robespierre, the younger brother of the Revolutionary leader Maximilien Robespierre.'''
                        #json_schema = {
                        #    "paragraphs": [],
                        #}
                        #example_output_json = {
                        #    "paragraph_1": '''Upon graduating in September 1785, Bonaparte was commissioned a second lieutenant in La Fère artillery regiment. He served in Valence and Auxonne until after the outbreak of the French Revolution in 1789.''',
                        #    "paragraph_2": '''Bonaparte was a fervent Corsican nationalist during this period. He asked for leave to join his mentor Pasquale Paoli, when Paoli was allowed to return to Corsica by the National Assembly. But Paoli had no sympathy for Napoleon, as he deemed his father a traitor for having deserted the cause of Corsican independence. He spent the early years of the Revolution in Corsica, fighting in a complex three-way struggle among royalists, revolutionaries, and Corsican nationalists.''',
                        #    "paragraph_3": '''Napoleon embraced the ideals of the Revolution, becoming a supporter of the Jacobins and joining the pro-French Corsican Republicans who opposed Paoli's policy and his aspirations to secede. He was given command over a battalion of volunteers and promoted to captain in the regular army in 1792, despite exceeding his leave of absence and leading a riot against French troops. When Corsica declared formal secession from France and requested the protection of the British government, Napoleon and his commitment to the French Revolution came into conflict with Paoli, who had decided to sabotage the Corsican contribution to the Expédition de Sardaigne by preventing a French assault on the Sardinian island La Maddalena.''',
                        #    "paragraph_4": '''Bonaparte and his family were compelled to flee to Toulon on the French mainland in June 1793 because of the split with Paoli. Although he was born "Napoleone Buonaparte", it was after this that Napoleon began styling himself "Napoléon Bonaparte". His family did not drop the name Buonaparte until 1796. The first known record of him signing his name as Bonaparte was at the age of 27 (in 1796).''',
                        #    "paragraph_5": '''In July 1793, Bonaparte published a pro-republican pamphlet, Le souper de Beaucaire (Supper at Beaucaire), which gained him the support of Augustin Robespierre, the younger brother of the Revolutionary leader Maximilien Robespierre.''',
                        #}
                        #example_narration2 = '''Despite a popular misconception to the contrary, nearly all educated Westerners of Columbus's time knew that the Earth is spherical, a concept that had been understood since antiquity. The techniques of celestial navigation, which uses the position of the Sun and the stars in the sky, had long been in use by astronomers and were beginning to be implemented by mariners. As far back as the 3rd century BC, Eratosthenes had correctly computed the circumference of the Earth by using simple geometry and studying the shadows cast by objects at two remote locations. In the 1st century BC, Posidonius confirmed Eratosthenes's results by comparing stellar observations at two separate locations. These measurements were widely known among scholars, but Ptolemy's use of the smaller, old-fashioned units of distance led Columbus to underestimate the size of the Earth by about a third. "Columbus map", drawn c. 1490 in the Lisbon mapmaking workshop of Bartholomew and Christopher Columbus Three cosmographical parameters determined the bounds of Columbus's enterprise: the distance across the ocean between Europe and Asia, which depended on the extent of the oikumene, i.e., the Eurasian land-mass stretching east–west between Spain and China; the circumference of the Earth; and the number of miles or leagues in a degree of longitude, which was possible to deduce from the theory of the relationship between the size of the surfaces of water and the land as held by the followers of Aristotle in medieval times. From Pierre d'Ailly's Imago Mundi (1410), Columbus learned of Alfraganus's estimate that a degree of latitude (equal to approximately a degree of longitude along the equator) spanned 56.67 Arabic miles (equivalent to 66.2 nautical miles, 122.6 kilometers or 76.2 mi), but he did not realize that this was expressed in the Arabic mile (about 1,830 meters or 1.14 mi) rather than the shorter Roman mile (about 1,480 m) with which he was familiar.'''
                        #example_output2 = '''Despite a popular misconception to the contrary, nearly all educated Westerners of Columbus's time knew that the Earth is spherical, a concept that had been understood since antiquity. The techniques of celestial navigation, which uses the position of the Sun and the stars in the sky, had long been in use by astronomers and were beginning to be implemented by mariners. As far back as the 3rd century BC, Eratosthenes had correctly computed the circumference of the Earth by using simple geometry and studying the shadows cast by objects at two remote locations.\n\nIn the 1st century BC, Posidonius confirmed Eratosthenes's results by comparing stellar observations at two separate locations. These measurements were widely known among scholars, but Ptolemy's use of the smaller, old-fashioned units of distance led Columbus to underestimate the size of the Earth by about a third.\n\n"Columbus map", drawn c. 1490 in the Lisbon mapmaking workshop of Bartholomew and Christopher Columbus Three cosmographical parameters determined the bounds of Columbus's enterprise: the distance across the ocean between Europe and Asia, which depended on the extent of the oikumene, i.e., the Eurasian land-mass stretching east–west between Spain and China; the circumference of the Earth; and the number of miles or leagues in a degree of longitude, which was possible to deduce from the theory of the relationship between the size of the surfaces of water and the land as held by the followers of Aristotle in medieval times.\n\nFrom Pierre d'Ailly's Imago Mundi (1410), Columbus learned of Alfraganus's estimate that a degree of latitude (equal to approximately a degree of longitude along the equator) spanned 56.67 Arabic miles (equivalent to 66.2 nautical miles, 122.6 kilometers or 76.2 mi), but he did not realize that this was expressed in the Arabic mile (about 1,830 meters or 1.14 mi) rather than the shorter Roman mile (about 1,480 m) with which he was familiar.'''
                        #example_output_json2 = {
                        #    "paragraph_1": '''Despite a popular misconception to the contrary, nearly all educated Westerners of Columbus's time knew that the Earth is spherical, a concept that had been understood since antiquity. The techniques of celestial navigation, which uses the position of the Sun and the stars in the sky, had long been in use by astronomers and were beginning to be implemented by mariners. As far back as the 3rd century BC, Eratosthenes had correctly computed the circumference of the Earth by using simple geometry and studying the shadows cast by objects at two remote locations.''',
                        #    "paragraph_2": '''In the 1st century BC, Posidonius confirmed Eratosthenes's results by comparing stellar observations at two separate locations. These measurements were widely known among scholars, but Ptolemy's use of the smaller, old-fashioned units of distance led Columbus to underestimate the size of the Earth by about a third.''',
                        #    "paragraph_3": '''"Columbus map", drawn c. 1490 in the Lisbon mapmaking workshop of Bartholomew and Christopher Columbus Three cosmographical parameters determined the bounds of Columbus's enterprise: the distance across the ocean between Europe and Asia, which depended on the extent of the oikumene, i.e., the Eurasian land-mass stretching east–west between Spain and China; the circumference of the Earth; and the number of miles or leagues in a degree of longitude, which was possible to deduce from the theory of the relationship between the size of the surfaces of water and the land as held by the followers of Aristotle in medieval times.''',
                        #    "paragraph_4": '''From Pierre d'Ailly's Imago Mundi (1410), Columbus learned of Alfraganus's estimate that a degree of latitude (equal to approximately a degree of longitude along the equator) spanned 56.67 Arabic miles (equivalent to 66.2 nautical miles, 122.6 kilometers or 76.2 mi), but he did not realize that this was expressed in the Arabic mile (about 1,830 meters or 1.14 mi) rather than the shorter Roman mile (about 1,480 m) with which he was familiar.''',
                        #}

                        #messages = [
                            #HumanMessage(
                            #    content=prompt + " Below is an example text:\n" + example_narration + "\nThe text divided will be:\n" + example_output
                            #),
                            #HumanMessage(
                            #    content=prompt + " Below is another example text:\n" + example_narration2 + "\nThe text divided will be:\n" + example_output2
                            #),
                            #HumanMessage(
                            #    content=prompt + " Respond considering the following JSON schema:"
                            #),
                            #HumanMessage(content=json.dumps(json_schema, indent=2)),
                            #HumanMessage(
                            #    content="\nNow, considering the schema, the output will be:"
                            #),
                            #HumanMessage(content=json.dumps(example_output_json, indent=2)),
                            #HumanMessage(
                            #    content="\nNow, considering the schema, " + prompt + "\nText:\n\n" + narrazione
                            #),
                        #]

                        start = time.time()

                        try:
                            #chat_model_response = chat_model(messages)
                            #events = chat_model_response.content
                            events = llm("Text:\n" + narrazione)
                            result = remove_tokens_to_match(events, narrazione)
                            json_result = create_json_from_text(result, 'paragraphs.json')
                            print("\nReading paragraphs from JSON result:\n")
                            print_lines_from_json(json_result, narrazione)
                            if "Cannot find a match by removing tokens." not in result:
                                paragraphs, sentences, mean_sentences = print_rows_and_paragraphs(result)
                                print("\nParagrafi: ", paragraphs, "\nFrasi: ", sentences,
                                      "\nMedia di frasi per paragrafo: ", round(mean_sentences, 1))
                            else:
                                print("\nDati su paragrafi e frasi non disponibili.\n")
                            jac = jaccard_similarity(result, narrazione)
                            if jac > max_jac:
                                max_jac = jac
                                events_res = result
                                best_prompt[titolo] = prompt
                        except ValueError:
                            print("Ollama did not respond. Run skipped.")
                            events = ''
                            events_res = ''
                        end = time.time()

                        tempi.append((end - start) / 60)

                        if events not in dizionario_stringhe.values():
                            # Aggiungi la stringa al dizionario
                            dizionario_stringhe[_] = events

                        if model not in dizionario_narrazioni:
                            dizionario_narrazioni[model] = {}

                        if model not in dizionario_tempo:
                            dizionario_tempo[model] = {}

                        if model not in dizionario_finale:
                            dizionario_finale[model] = {}

                        print("\n")
                        print("Finita iterazione #" + str(_) + " sul testo \"" + titolo + "\" con modello " + model)
                        print("\n")

                    if max_jac < jaccard_threshold:
                        for jprompt in jaccard_systemPrompt:
                            for n in range(iterazioniSuOgniNarrazione):

                                print("Inizio sottoiterazione Jaccard #" + str(n) + " sul testo \"" + titolo + "\" con modello " + model + "\nsystemPrompt: " + jprompt)
                                print("\n")

                                llm = Ollama(
                                    model=model,
                                    system=jprompt,
                                #    format='json',
                                    num_ctx=4096,
                                    temperature=0.01,
                                    #repeat_last_n=0,
                                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                                )

                                #chat_model = ChatOllama(
                                #    model=model,
                                #    format="json",
                                #    #system=prompt,
                                #    num_ctx=4096,
                                #    temperature=0.01,
                                #    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                                #)

                                #example_narration = '''Upon graduating in September 1785, Bonaparte was commissioned a second lieutenant in La Fère artillery regiment. He served in Valence and Auxonne until after the outbreak of the French Revolution in 1789. Bonaparte was a fervent Corsican nationalist during this period. He asked for leave to join his mentor Pasquale Paoli, when Paoli was allowed to return to Corsica by the National Assembly. But Paoli had no sympathy for Napoleon, as he deemed his father a traitor for having deserted the cause of Corsican independence. He spent the early years of the Revolution in Corsica, fighting in a complex three-way struggle among royalists, revolutionaries, and Corsican nationalists. Napoleon embraced the ideals of the Revolution, becoming a supporter of the Jacobins and joining the pro-French Corsican Republicans who opposed Paoli's policy and his aspirations to secede. He was given command over a battalion of volunteers and promoted to captain in the regular army in 1792, despite exceeding his leave of absence and leading a riot against French troops. When Corsica declared formal secession from France and requested the protection of the British government, Napoleon and his commitment to the French Revolution came into conflict with Paoli, who had decided to sabotage the Corsican contribution to the Expédition de Sardaigne by preventing a French assault on the Sardinian island La Maddalena. Bonaparte and his family were compelled to flee to Toulon on the French mainland in June 1793 because of the split with Paoli. Although he was born "Napoleone Buonaparte", it was after this that Napoleon began styling himself "Napoléon Bonaparte". His family did not drop the name Buonaparte until 1796. The first known record of him signing his name as Bonaparte was at the age of 27 (in 1796). In July 1793, Bonaparte published a pro-republican pamphlet, Le souper de Beaucaire (Supper at Beaucaire), which gained him the support of Augustin Robespierre, the younger brother of the Revolutionary leader Maximilien Robespierre.'''
                                #example_output = '''Upon graduating in September 1785, Bonaparte was commissioned a second lieutenant in La Fère artillery regiment. He served in Valence and Auxonne until after the outbreak of the French Revolution in 1789.\n\nBonaparte was a fervent Corsican nationalist during this period. He asked for leave to join his mentor Pasquale Paoli, when Paoli was allowed to return to Corsica by the National Assembly. But Paoli had no sympathy for Napoleon, as he deemed his father a traitor for having deserted the cause of Corsican independence. He spent the early years of the Revolution in Corsica, fighting in a complex three-way struggle among royalists, revolutionaries, and Corsican nationalists.\n\nNapoleon embraced the ideals of the Revolution, becoming a supporter of the Jacobins and joining the pro-French Corsican Republicans who opposed Paoli's policy and his aspirations to secede. He was given command over a battalion of volunteers and promoted to captain in the regular army in 1792, despite exceeding his leave of absence and leading a riot against French troops. When Corsica declared formal secession from France and requested the protection of the British government, Napoleon and his commitment to the French Revolution came into conflict with Paoli, who had decided to sabotage the Corsican contribution to the Expédition de Sardaigne by preventing a French assault on the Sardinian island La Maddalena.\n\nBonaparte and his family were compelled to flee to Toulon on the French mainland in June 1793 because of the split with Paoli. Although he was born "Napoleone Buonaparte", it was after this that Napoleon began styling himself "Napoléon Bonaparte". His family did not drop the name Buonaparte until 1796. The first known record of him signing his name as Bonaparte was at the age of 27 (in 1796).\n\nIn July 1793, Bonaparte published a pro-republican pamphlet, Le souper de Beaucaire (Supper at Beaucaire), which gained him the support of Augustin Robespierre, the younger brother of the Revolutionary leader Maximilien Robespierre.'''
                                #json_schema = {
                                #    "paragraphs": [],
                                #}
                                #example_output_json = {
                                #    "paragraph_1": '''Upon graduating in September 1785, Bonaparte was commissioned a second lieutenant in La Fère artillery regiment. He served in Valence and Auxonne until after the outbreak of the French Revolution in 1789.''',
                                #    "paragraph_2": '''Bonaparte was a fervent Corsican nationalist during this period. He asked for leave to join his mentor Pasquale Paoli, when Paoli was allowed to return to Corsica by the National Assembly. But Paoli had no sympathy for Napoleon, as he deemed his father a traitor for having deserted the cause of Corsican independence. He spent the early years of the Revolution in Corsica, fighting in a complex three-way struggle among royalists, revolutionaries, and Corsican nationalists.''',
                                #    "paragraph_3": '''Napoleon embraced the ideals of the Revolution, becoming a supporter of the Jacobins and joining the pro-French Corsican Republicans who opposed Paoli's policy and his aspirations to secede. He was given command over a battalion of volunteers and promoted to captain in the regular army in 1792, despite exceeding his leave of absence and leading a riot against French troops. When Corsica declared formal secession from France and requested the protection of the British government, Napoleon and his commitment to the French Revolution came into conflict with Paoli, who had decided to sabotage the Corsican contribution to the Expédition de Sardaigne by preventing a French assault on the Sardinian island La Maddalena.''',
                                #    "paragraph_4": '''Bonaparte and his family were compelled to flee to Toulon on the French mainland in June 1793 because of the split with Paoli. Although he was born "Napoleone Buonaparte", it was after this that Napoleon began styling himself "Napoléon Bonaparte". His family did not drop the name Buonaparte until 1796. The first known record of him signing his name as Bonaparte was at the age of 27 (in 1796).''',
                                #    "paragraph_5": '''In July 1793, Bonaparte published a pro-republican pamphlet, Le souper de Beaucaire (Supper at Beaucaire), which gained him the support of Augustin Robespierre, the younger brother of the Revolutionary leader Maximilien Robespierre.''',
                                #}
                                # example_narration2 = '''Despite a popular misconception to the contrary, nearly all educated Westerners of Columbus's time knew that the Earth is spherical, a concept that had been understood since antiquity. The techniques of celestial navigation, which uses the position of the Sun and the stars in the sky, had long been in use by astronomers and were beginning to be implemented by mariners. As far back as the 3rd century BC, Eratosthenes had correctly computed the circumference of the Earth by using simple geometry and studying the shadows cast by objects at two remote locations. In the 1st century BC, Posidonius confirmed Eratosthenes's results by comparing stellar observations at two separate locations. These measurements were widely known among scholars, but Ptolemy's use of the smaller, old-fashioned units of distance led Columbus to underestimate the size of the Earth by about a third. "Columbus map", drawn c. 1490 in the Lisbon mapmaking workshop of Bartholomew and Christopher Columbus Three cosmographical parameters determined the bounds of Columbus's enterprise: the distance across the ocean between Europe and Asia, which depended on the extent of the oikumene, i.e., the Eurasian land-mass stretching east–west between Spain and China; the circumference of the Earth; and the number of miles or leagues in a degree of longitude, which was possible to deduce from the theory of the relationship between the size of the surfaces of water and the land as held by the followers of Aristotle in medieval times. From Pierre d'Ailly's Imago Mundi (1410), Columbus learned of Alfraganus's estimate that a degree of latitude (equal to approximately a degree of longitude along the equator) spanned 56.67 Arabic miles (equivalent to 66.2 nautical miles, 122.6 kilometers or 76.2 mi), but he did not realize that this was expressed in the Arabic mile (about 1,830 meters or 1.14 mi) rather than the shorter Roman mile (about 1,480 m) with which he was familiar.'''
                                # example_output2 = '''Despite a popular misconception to the contrary, nearly all educated Westerners of Columbus's time knew that the Earth is spherical, a concept that had been understood since antiquity. The techniques of celestial navigation, which uses the position of the Sun and the stars in the sky, had long been in use by astronomers and were beginning to be implemented by mariners. As far back as the 3rd century BC, Eratosthenes had correctly computed the circumference of the Earth by using simple geometry and studying the shadows cast by objects at two remote locations.\n\nIn the 1st century BC, Posidonius confirmed Eratosthenes's results by comparing stellar observations at two separate locations. These measurements were widely known among scholars, but Ptolemy's use of the smaller, old-fashioned units of distance led Columbus to underestimate the size of the Earth by about a third.\n\n"Columbus map", drawn c. 1490 in the Lisbon mapmaking workshop of Bartholomew and Christopher Columbus Three cosmographical parameters determined the bounds of Columbus's enterprise: the distance across the ocean between Europe and Asia, which depended on the extent of the oikumene, i.e., the Eurasian land-mass stretching east–west between Spain and China; the circumference of the Earth; and the number of miles or leagues in a degree of longitude, which was possible to deduce from the theory of the relationship between the size of the surfaces of water and the land as held by the followers of Aristotle in medieval times.\n\nFrom Pierre d'Ailly's Imago Mundi (1410), Columbus learned of Alfraganus's estimate that a degree of latitude (equal to approximately a degree of longitude along the equator) spanned 56.67 Arabic miles (equivalent to 66.2 nautical miles, 122.6 kilometers or 76.2 mi), but he did not realize that this was expressed in the Arabic mile (about 1,830 meters or 1.14 mi) rather than the shorter Roman mile (about 1,480 m) with which he was familiar.'''
                                # example_output_json2 = {
                                #    "paragraph_1": '''Despite a popular misconception to the contrary, nearly all educated Westerners of Columbus's time knew that the Earth is spherical, a concept that had been understood since antiquity. The techniques of celestial navigation, which uses the position of the Sun and the stars in the sky, had long been in use by astronomers and were beginning to be implemented by mariners. As far back as the 3rd century BC, Eratosthenes had correctly computed the circumference of the Earth by using simple geometry and studying the shadows cast by objects at two remote locations.''',
                                #    "paragraph_2": '''In the 1st century BC, Posidonius confirmed Eratosthenes's results by comparing stellar observations at two separate locations. These measurements were widely known among scholars, but Ptolemy's use of the smaller, old-fashioned units of distance led Columbus to underestimate the size of the Earth by about a third.''',
                                #    "paragraph_3": '''"Columbus map", drawn c. 1490 in the Lisbon mapmaking workshop of Bartholomew and Christopher Columbus Three cosmographical parameters determined the bounds of Columbus's enterprise: the distance across the ocean between Europe and Asia, which depended on the extent of the oikumene, i.e., the Eurasian land-mass stretching east–west between Spain and China; the circumference of the Earth; and the number of miles or leagues in a degree of longitude, which was possible to deduce from the theory of the relationship between the size of the surfaces of water and the land as held by the followers of Aristotle in medieval times.''',
                                #    "paragraph_4": '''From Pierre d'Ailly's Imago Mundi (1410), Columbus learned of Alfraganus's estimate that a degree of latitude (equal to approximately a degree of longitude along the equator) spanned 56.67 Arabic miles (equivalent to 66.2 nautical miles, 122.6 kilometers or 76.2 mi), but he did not realize that this was expressed in the Arabic mile (about 1,830 meters or 1.14 mi) rather than the shorter Roman mile (about 1,480 m) with which he was familiar.''',
                                # }

                                #messages = [
                                    #HumanMessage(
                                    #    content=prompt + " Below is an example text:\n" + example_narration + "\nThe text divided will be:\n" + example_output
                                    #),
                                    # HumanMessage(
                                    #    content=prompt + " Below is another example text:\n" + example_narration2 + "\nThe text divided will be:\n" + example_output2
                                    # ),
                                    # HumanMessage(
                                    #    content="\nThe output format must follow the following JSON schema:"
                                    # ),
                                    # HumanMessage(content=json.dumps(json_schema, indent=2)),
                                    # HumanMessage(
                                    #    content="\nNow, considering the schema, the output will be:"
                                    # ),
                                    # HumanMessage(content=json.dumps(example_output_json, indent=2)),
                                    #HumanMessage(
                                    #    content=prompt + " Respond considering the following JSON schema:"
                                    #),
                                    #HumanMessage(content=json.dumps(json_schema, indent=2)),
                                    #HumanMessage(
                                    #    content="\nNow, considering the schema, " + prompt + "\nText:\n\n" + narrazione
                                    #),
                                #]

                                start = time.time()
                                try:
                                    #chat_model_response = chat_model(messages)
                                    #jaccard_events = chat_model_response.content
                                    jaccard_events = llm("Text:\n" + narrazione)
                                    result2 = remove_tokens_to_match(jaccard_events, narrazione)
                                    json_result2 = create_json_from_text(result2, 'paragraphs2.json')
                                    print("\nReading paragraphs from JSON result2:\n")
                                    print_lines_from_json(json_result2, narrazione)
                                    if "Cannot find a match by removing tokens." not in result2:
                                        paragraphs, sentences, mean_sentences = print_rows_and_paragraphs(result2)
                                        print("\nParagrafi: ", paragraphs, "\nFrasi: ", sentences,
                                              "\nMedia di frasi per paragrafo: ", round(mean_sentences, 1))
                                    else:
                                        print("\nDati su paragrafi e frasi non disponibili.\n")
                                    jac2 = jaccard_similarity(result2, narrazione)
                                    if jac2 > max_jac:
                                        max_jac = jac2
                                        events_res = result2
                                        best_prompt[titolo] = jprompt

                                except ValueError:
                                    print("Ollama did not respond. Run skipped.")
                                    jaccard_events = ''
                                    events_res = ''
                                end = time.time()

                                tempi.append((end - start) / 60)

                                if jaccard_events not in dizionario_stringhe.values():
                                    # Aggiungi la stringa al dizionario
                                    dizionario_stringhe[n] = jaccard_events

                                if model not in dizionario_narrazioni:
                                    dizionario_narrazioni[model] = {}

                                if model not in dizionario_finale:
                                    dizionario_finale[model] = {}

                                if model not in dizionario_tempo:
                                    dizionario_tempo[model] = {}

                                print("\n")
                                print("Finita sottoiterazione Jaccard #" + str(n) + " sul testo \"" + titolo + "\" con modello " + model)
                                print("\n")

                dizionario_narrazioni[model][titolo] = dizionario_stringhe
                dizionario_tempo[model][titolo] = tempi
                print("L'output scelto con Jaccard", round(max_jac, 2), " e prompt \"" + best_prompt[titolo] + "\" e':\n")
                print(events_res)
                dizionario_finale[model][titolo] = events_res

    # Salva il dizionario e metadata in due file json
    save_dict_to_file(dizionario_narrazioni, 'dizionario_narrazioni_output.json')
    save_dict_to_file(dizionario_tempo, 'dizionario_tempo.json')
    save_dict_to_file(dizionario_finale, 'dizionario_finale.json')
    save_dict_to_file(best_prompt, 'best_prompt.json')
    metadata = {}
    metadata["iterazioni"] = iterazioniSuOgniNarrazione
    metadata["narrazioni"] = narrazioni
    metadata["titoliNarrazione"] = titoliNarrazione
    save_dict_to_file(metadata, 'metadata.json')
    print("Risultati salvati correttamente. Processo completato.")