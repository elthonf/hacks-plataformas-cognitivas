import json
with open('mykey.json') as json_file:
    mykey = json.load(json_file)


from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
authenticator = IAMAuthenticator(mykey["key"])


visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    authenticator=authenticator)
visual_recognition.set_service_url(mykey["url"])

#desabilita SSL
#visual_recognition.disable_SSL_verification()

with open('../../../datasets/imagens/lions/imagem_test1.jpg', 'rb') as one_image_file:
    classes = visual_recognition.classify(
        images_file=one_image_file,
        threshold=0.6,
        classifier_ids='default').get_result()


#with open('datasets/imagens/lions/imagem_test1.jpg', 'rb') as one_image_file:
#    classes = visual_recognition.classify(
#        images_file=one_image_file,
#        threshold=0.6,
#        classifier_ids='explicit').get_result()

print(json.dumps(classes, indent=2))
