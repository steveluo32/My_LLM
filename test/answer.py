from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from environment import *

# Initialize OpenAI Chat model
chat = ChatOpenAI(model="gpt-4", temperature=0.1)

# Question to be asked
question = "hci projects"

# Initialize dictionaries and lists
content_dict = {}
path_dict = {}
num_list = []
path_list = []

# Read content file and populate content_dict
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        key, value = line.split(' ', 1)
        content_dict[key] = value

# Read path file and populate path_dict
with open(path, 'r') as file:
    for line in file:
        line = line.strip()
        key, value = line.split(' ', 1)
        path_dict[key] = value

# Function to find key by value content
def find_key_by_value_content(dictionary, search_str):
    for key, value in dictionary.items():
        if search_str in value:
            return key
    return None

# Sample response lines (replace with actual response lines from your system)
response_lines = ['./data/scraping_data/cis.unimelb.edu.au/hci/projects/digital-game-ethics/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/pholiota-unlocked/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/vr-climate-change/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/changing-views/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/death/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/vrtherapy/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/ageing-avatars/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/intimacy/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/human-centred-agent-learning/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/human-agent-collaboration/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/digital-domesticity/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/orygen-virtual-world/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/cognitive-interaction/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/meal-times/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/near-infrared-spectroscopy/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/augmented-fitness/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/learners/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/search/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/interactive-spaces/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/video-calls/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/spinallog/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/indigenous-knowledge/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/adaptive-learning-technologies/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/hospital/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/robot-assisted-learning-and-rehabilitation/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/digital-technologies-for-urban-nature-sites/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/digitalemotionregulation/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/ifish/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/onebody/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/smartphones/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/facespace/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/biometric-mirror/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/cognition-aware-systems/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/sore/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/virtual-co-presence/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/sgw/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/citizenheritage/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/encounters/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/music-algorithmic-recommendation/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/complexdata/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/social-and-domestic-drones/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/armsleeve/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/vitamind/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/mobilefieldwork/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/enrichmentoldage/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/child-of-now/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/ambivalence/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/crowdsourcing/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/completed/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/xr-for-human-robot-interaction/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/handlog/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/physio-education/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/hybrid-digital-boardgames/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/strategic-planning-games/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/spectating-esports/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/socialrobots/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/warhammer/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/ccis/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/zoos/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/reading-on-ubiquitous-devices/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/interactive-displays/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/peerreview/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/scale/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/deceptive-ai/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/thismymob-establishing-digital-land-rights/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/digital-commemoration/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/social-play/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/sociophysical/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/personal-sensing/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/teleconsultation/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/zoos#contact/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/ageing/content.txt', './data/scraping_data/cis.unimelb.edu.au/hci/projects/insertables/content.txt']
response_lines = ['scraping data cis unimelb edu au people students david adams content txt']

# Find keys by value content and populate num_list
for line in response_lines:
    num = find_key_by_value_content(content_dict, line)
    if num:
        num_list.append(num)

# Populate path_list with corresponding paths
for num in num_list:
    path = os.path.join(data_path, path_dict[num])
    path_list.append(path)

# Function to read content from files in path_list
def read_files(file_paths):
    content = ""
    for path in file_paths:
        with open(path, 'r') as file:
            content += file.read() + "\n"
    return content

# Check if path_list is not empty and process accordingly
if path_list:
    file_content = read_files(path_list)
    messages = [HumanMessage(content=file_content + "\n" + question)]
else:
    messages = [HumanMessage(content=question)]

# Get response from the chat model
response = chat.invoke(messages)

# Print the response
print(response.content)
