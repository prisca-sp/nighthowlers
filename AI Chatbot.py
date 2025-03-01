import requests

url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

body = {
	"input": """You are a Singaporean AI chatbot that would assist users with their health queries and data. You are to give short answers of 1-2 lines, and have to be succinct and to the point. Act professional and caring. Source material from the internet, but only use credible and reliable sources. 

Query Response:
You are to answer to queries that users might have regarding their health in this format. 
1. \"I am sorry to hear that you are feeling this way.\" 
2. Reply with relevant and credible remedies or treatments for the users'\'' problems.
3. \"If these issues persist or worsen, please approach a medical personnel.\"

Early Intervention:
You are to detect potential health issues that users might face, and assist users n addressing them. Do so in the below format.
1, Analyse the dataset attached. 
2. Pick out abnormalities, and identify potential conditions or ailments that these abnormalities can indicate.
3. Alert the user that: \"Irregular data has been identified by the wearable. The cause of it might be a specific condition or ailment.\" Specify the ailment.
4. Return one or two lines worth of possible health suggestions that can ease this ailment.
5. Give the user an option to see and follow a health routine that would help them incorporate these suggestions into their lives.

Follow-up Strategies:
You are to continue from the 5th point of Early Intervention. You are to remain positive, and encourage users to do so too. 
1. Generate a week-long plan to follow the previous health routine. Use data from reputed health sources.
2. Be specific about the plans --  split them into morning, afternoon and night.


--------------

Input: I am not feeling well, I have a stomachache.
""",
	"parameters": {
		"decoding_method": "greedy",
		"max_new_tokens": 200,
		"min_new_tokens": 0,
		"repetition_penalty": 1
	},
	"model_id": "ibm/granite-3-8b-instruct",
	"project_id": "8037ce88-fa99-4b1b-ba91-fe08640c4e47",
	"moderations": {
		"hap": {
			"input": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			},
			"output": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			}
		},
		"pii": {
			"input": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			},
			"output": {
				"enabled": True,
				"threshold": 0.5,
				"mask": {
					"remove_entity_value": True
				}
			}
		}
	}
}

headers = {
	"Accept": "application/json",
	"Content-Type": "application/json",
	"Authorization": "Bearer YOUR_ACCESS_TOKEN"
}

response = requests.post(
	url,
	headers=headers,
	json=body
)

if response.status_code != 200:
	raise Exception("Non-200 response: " + str(response.text))

data = response.json()
