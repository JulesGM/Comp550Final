###

# Clone the bookcorpus repository
mkdir bookcorpus-repo
git clone https://github.com/soskek/bookcorpus bookcorpus-repo

# Install requirements for bookcorpus
python -m pip install -r bookcorpus-repo/requirements.txt

# Get the list of book json
python -u tmp-bookcorpus/download_list.py > url_list.jsonl




