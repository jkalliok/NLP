from plotWindow import plotWindow
import matplotlib.pyplot as plt
import numpy as np
import json

with open('database.json', 'r') as f:
    database = json.load(f)

def split(word):
    return [char for char in word]
labels = split("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

pw = plotWindow()

#xPoints = []
#for snippet in database['snippets']:
#    xPoints.append(database['snippets'][snippet]['count'])

groups = []
for cluster in database['clusters']['SearchCount']:
    group = []
    for letter in database['clusters']['SearchCount'][cluster]:
        group.append(database['snippets'][letter]['count'])
    groups.append(group)

fig = plt.figure()
plt.xlabel("Count")
plt.ylabel("")
xPoints = groups[0]
plt.plot(xPoints, [1]*len(xPoints), '.b')

xPoints = groups[1]
plt.plot(xPoints, [1]*len(xPoints), '.r')

xPoints = groups[2]
plt.plot(xPoints, [1]*len(xPoints), '.g')

xPoints = groups[3]
plt.plot(xPoints, [1]*len(xPoints), '.c')

search_table = [[" ".join(database["clusters"]["SearchCount"]["a"]).upper(), " ".join(database["clusters"]["SearchCount"]["b"]).upper(), " ".join(database["clusters"]["SearchCount"]["c"]).upper(), " ".join(database["clusters"]["SearchCount"]["d"]).upper()]]
search_table = plt.table(search_table, loc="top")
search_table.auto_set_font_size(False)
search_table.set_fontsize(7)
search_table.scale(1,2)
pw.addPlot("Google Search Count", fig)

fig = plt.figure()
plt.imshow(database['cosines']['w2v'])
plt.xticks(range(26), labels)
plt.yticks(range(26), labels)
pw.addPlot("Cosines - Word2Vector", fig)

fig = plt.figure()
plt.imshow(database['cosines']['ft'])
plt.xticks(range(26), labels)
plt.yticks(range(26), labels)
pw.addPlot("Cosines - FastText", fig)

fig = plt.figure()
plt.imshow(database['cosines']['tfidf'])
plt.xticks(range(26), labels)
plt.yticks(range(26), labels)
pw.addPlot("Cosines - TfIdf", fig)

fig = plt.figure()
plt.imshow(database['fuzzywuzzy'])
plt.xticks(range(26), labels)
plt.yticks(range(26), labels)
pw.addPlot("FuzzyWuzzy", fig)



clusters = database['clusters']
table = [[" ".join(cl).upper() for cl in clusters['w2v']], [" ".join(cl).upper() for cl in clusters['ft']], [" ".join(cl).upper() for cl in clusters['tfidf']], [" ".join(cl).upper() for cl in clusters['fw']]]
fig = plt.figure(tight_layout=True, frameon=False)
plt.xticks([])
plt.yticks([])
table = plt.table(table, rowLabels = ["Word2vector", "FastText", "TfIdf", "Fuzzywuzzy"], loc="center")
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(0.75,2)
pw.addPlot("Clusters", fig)


pw.show()