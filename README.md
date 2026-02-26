<div align="center">
    <h1>Automatic Summarization System</h1>
    <h3>Authors: Christian Faccio, Elena Lorite Acosta, Rebeca Piñol Galera, Paula Frías Arroyo</h3>
    <h5>Emails: christianfaccio@outlook.it, elenalorite@gmail.com, rpg80@alu.ua.es, pfa13@alu.ua.es</h4>
    <h5>Github: <a href="https://github.com/christianfaccio" target="_blank">christianfaccio</a>, <a href="https://github.com/elorite" target"_blank">elorite</a>, <a href="https://github.com/rebeca342" target"_blank">rebeca342</a>, <a href="https://github.com/pfa13" target"_blank">pfa13</a></h5>
    <h6></h6>
</div>

--- 

### How to run the code

First of all, create a virtual environment and install the dependencies:
```
uv venv nlp --python 3.11
source nlp/bin/activate
uv pip install -r requirements.txt
```

Then, from the `src/` folder run one or both the information retrieval scripts:
```
cd src
uv run ir.py
uv run nir.py
```

After that, you should be able to see in the `output/` folder the json files containing the structured output. Now you can run the generation script:
```
uv run generation.py
```

and you should see the generated summaries in txt format in the `output/generated_summaries` folder.

>[!note]
>Consider that if you have not already downloaded the models it could take some time to finish.

Finally, you can run the evaluation script to get the score of the obtained summaries:
```
uv run evaluate.py
```