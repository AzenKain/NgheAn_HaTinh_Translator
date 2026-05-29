import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
DICTIONARY_PATH = Path(os.getenv("DICTIONARY_PATH", "nghe_an_dict_qwen3_cleaned.json"))

# Setup OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Global variables for RAG
hash_map = {}
collection = None
embedding_model = None
all_dict_keys = []

def init_translator():
    global hash_map, collection, embedding_model, all_dict_keys
    
    if not DICTIONARY_PATH.exists():
        raise FileNotFoundError(f"Dictionary not found at {DICTIONARY_PATH}")
        
    print(f"[RAG] Loading dictionary from {DICTIONARY_PATH}...")
    with DICTIONARY_PATH.open("r", encoding="utf-8") as f:
        dictionary_data = json.load(f)
        
    # Build exact matching hash map
    def norm_text(s):
        return re.sub(r"\s+", " ", str(s).lower().strip())

    def build_context(entry):
        parts = []
        meaning = entry.get("meaning", "")
        example = entry.get("example", "")
        example_trans = entry.get("example_translate", "")
        note = entry.get("note", "")
        source_quality = entry.get("source_quality", "")

        if meaning:
            parts.append(f"Nghĩa: {meaning}")
        if example:
            parts.append(f"Ví dụ: {example}")
        if example_trans:
            parts.append(f"Dịch ví dụ: {example_trans}")
        if note:
            parts.append(f"Ghi chú: {note}")
        if source_quality:
            parts.append(f"Độ tin cậy nguồn: {source_quality}")

        return " | ".join(parts)

    hash_map = {}
    for entry in dictionary_data:
        context_str = build_context(entry)
        keyword = norm_text(entry.get("keyword", ""))
        if keyword:
            hash_map[keyword] = context_str

        for alias in entry.get("aliases", []):
            alias = norm_text(alias)
            if alias:
                hash_map[alias] = context_str

    print(f"[RAG] Exact match map built with {len(hash_map)} keys.")
    all_dict_keys = list(hash_map.keys())

    # Initialize ChromaDB and SentenceTransformer
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print("[RAG] Failed to import ChromaDB or SentenceTransformer. Please wait for dependencies installation.")
        raise e

    print("[RAG] Loading SentenceTransformer model 'BAAI/bge-m3'...")
    embedding_model = SentenceTransformer("BAAI/bge-m3")
    
    # Use persistent Chroma client
    db_path = Path("build/chroma_db")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    
    # Calculate the expected number of entries from JSON dictionary
    expected_docs = 0
    for entry in dictionary_data:
        text_to_index = [entry.get("keyword", "")] + entry.get("aliases", [])
        for text in text_to_index:
            if norm_text(text):
                expected_docs += 1
                
    collection = chroma_client.get_or_create_collection(name="nghe_an_dic")
    
    # Rebuild collection if empty or if count mismatch (dictionary file updated)
    if collection.count() != expected_docs:
        print(f"[RAG] Vector DB count ({collection.count()}) mismatches expected dictionary entries ({expected_docs}). Rebuilding...")
        try:
            chroma_client.delete_collection(name="nghe_an_dic")
        except Exception:
            pass
        collection = chroma_client.get_or_create_collection(name="nghe_an_dic")
        
        documents = []
        metadatas = []
        ids = []

        for i, entry in enumerate(dictionary_data):
            text_to_index = [entry.get("keyword", "")] + entry.get("aliases", [])

            for j, text in enumerate(text_to_index):
                text = norm_text(text)
                if not text:
                    continue

                documents.append(text)
                metadatas.append({
                    "keyword": entry.get("keyword", ""),
                    "definition": entry.get("meaning", ""),
                    "example": entry.get("example", ""),
                    "example_translate": entry.get("example_translate", ""),
                    "note": entry.get("note", ""),
                    "source_quality": entry.get("source_quality", "")
                })
                ids.append(f"id_{i}_{j}")

        print(f"[RAG] Encoding {len(documents)} entries (this may take a moment)...")
        embeddings = embedding_model.encode(documents).tolist()
        
        # Batch addition in case it is too large
        batch_size = 500
        for idx in range(0, len(documents), batch_size):
            collection.add(
                embeddings=embeddings[idx:idx+batch_size],
                documents=documents[idx:idx+batch_size],
                metadatas=metadatas[idx:idx+batch_size],
                ids=ids[idx:idx+batch_size]
            )
        print("[RAG] Vector database ready.")
    else:
        print(f"[RAG] Loaded existing vector database with {collection.count()} entries.")

def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def get_closest_fuzzy_match(word, possibilities):
    word_len = len(word)
    if word_len <= 2:
        return None
        
    best_match = None
    min_dist = 999
    
    # We restrict max allowable distance based on word length:
    # 1 typo allowed for <= 4 chars, 2 typos allowed for longer words
    max_allowable_dist = 1 if word_len <= 4 else 2
    
    for pos in possibilities:
        if abs(len(pos) - word_len) > max_allowable_dist:
            continue
            
        dist = edit_distance(word, pos)
        if dist <= max_allowable_dist and dist < min_dist:
            min_dist = dist
            best_match = pos
            
    return best_match

def get_context_with_ngram(text, rag_top_k=3, max_distance=0.35):
    if not embedding_model or not collection:
        return "", []
        
    def norm_text(s):
        return re.sub(r"\s+", " ", str(s).lower().strip())
        
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    found_contexts = []
    found_keys = set()
    matched_indices = set()
    matched_terms = [] # To return to UI for highlight

    # Match using n-gram exact match and fuzzy match
    import difflib
    max_n = max((len(k.split()) for k in hash_map.keys()), default=3)

    for n in range(max_n, 0, -1):
        for i in range(len(words) - n + 1):
            if any(idx in matched_indices for idx in range(i, i + n)):
                continue

            gram = " ".join(words[i:i + n])
            gram_norm = norm_text(gram)

            if gram_norm in hash_map:
                found_contexts.append(f"- EXACT {gram.upper()}: {hash_map[gram_norm]}")
                found_keys.add(gram_norm)
                matched_indices.update(range(i, i + n))
                matched_terms.append({
                    "term": gram,
                    "type": "exact",
                    "info": hash_map[gram_norm]
                })
            else:
                # Fuzzy matching using custom Levenshtein edit distance logic (typo correction)
                if len(gram_norm) > 2 and all_dict_keys:
                    closest = get_closest_fuzzy_match(gram_norm, all_dict_keys)
                    if closest:
                        found_contexts.append(f"- GẦN GIỐNG VỚI {gram.upper()} (Có thể là {closest.upper()}): {hash_map[closest]}")
                        found_keys.add(closest)
                        matched_indices.update(range(i, i + n))
                        matched_terms.append({
                            "term": f"{gram} (~ {closest})",
                            "type": "fuzzy",
                            "info": hash_map[closest]
                        })

    # Match using semantic search
    query_emb = embedding_model.encode([text]).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=rag_top_k,
        include=["documents", "metadatas", "distances"]
    )

    if results and "documents" in results and len(results["documents"][0]) > 0:
        for i in range(len(results["documents"][0])):
            match_word = results["documents"][0][i]
            distance = results["distances"][0][i]
            meta = results["metadatas"][0][i]

            match_word_norm = norm_text(match_word)
            if match_word_norm in found_keys:
                continue

            if distance > max_distance:
                continue

            parts = []
            if meta.get("definition"):
                parts.append(f"Nghĩa: {meta['definition']}")
            if meta.get("example"):
                parts.append(f"Ví dụ: {meta['example']}")
            if meta.get("example_translate"):
                parts.append(f"Dịch ví dụ: {meta['example_translate']}")
            if meta.get("note"):
                parts.append(f"Ghi chú: {meta['note']}")
            if meta.get("source_quality"):
                parts.append(f"Độ tin cậy nguồn: {meta['source_quality']}")

            info = " | ".join(parts)
            found_contexts.append(f"- GẦN GIỐNG {match_word.upper()}: {info}")
            matched_terms.append({
                "term": match_word,
                "type": "semantic",
                "info": info
            })

    return "\n".join(found_contexts), matched_terms

def clean_translation(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    prefixes = [
        "Bản dịch:",
        "Tiếng Việt phổ thông:",
        "Câu tiếng Việt phổ thông:",
        "Kết quả:",
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    text = text.strip(" \n\r\t\"'“”")
    return text

def translate_nt_text(dialect_text: str):
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        return "Lỗi: Vui lòng cấu hình OPENROUTER_API_KEY trong file .env để thực hiện dịch.", []

    context, matched_terms = get_context_with_ngram(dialect_text)

    system_prompt = (
        "Bạn là bộ chuyển đổi phương ngữ Nghệ An/Nghệ Tĩnh sang tiếng Việt phổ thông.\n"
        "Nhiệm vụ: chuyển câu gốc sang tiếng Việt phổ thông tự nhiên, giữ nguyên ý, sắc thái và quan hệ xưng hô.\n"
        "\n"
        "Quy tắc xử lý lỗi chính tả và nhận diện âm thanh (ASR):\n"
        "- Câu đầu vào được thu âm và nhận diện bằng ASR nên có thể sai chính tả, thiếu dấu hoặc sai dấu do đặc thù giọng Nghệ Tĩnh (ví dụ: phát âm hỏi/ngã/nặng lẫn lộn, tr/ch, d/r/gi lẫn lộn, từ đồng âm khác cách viết).\n"
        "- Hãy tự động suy luận từ đúng dựa trên ngữ cảnh xung quanh của câu kết hợp với các từ gợi ý 'GẦN GIỐNG VỚI' trong từ điển tham khảo.\n"
        "- Ví dụ: ASR nhận diện nhầm 'đít lác' thành 'đít lách' hay 'đét lác', bạn cần tự khôi phục lại từ đúng nghĩa là 'đít lác' (hết tiền) và dịch chính xác.\n"
        "\n"
        "Quy tắc đầu ra:\n"
        "- Chỉ trả về duy nhất bản dịch tiếng Việt phổ thông.\n"
        "- Không giải thích.\n"
        "- Không phân tích.\n"
        "- Không dùng thẻ <think>.\n"
        "- Không lặp lại câu gốc.\n"
        "- Không thêm thông tin mới ngoài ý của câu gốc.\n"
        "\n"
        "Quy tắc chống dịch ảo:\n"
        "- Không được tự sáng tạo từ mới.\n"
        "- Không được thay đổi các từ/cụm từ tiếng Việt phổ thông vốn đã đúng nghĩa.\n"
        "- Nếu một cụm đã là tiếng phổ thông tự nhiên, hãy giữ nguyên cụm đó.\n"
        "- Chỉ thay thế các từ/cụm thật sự là phương ngữ Nghệ An/Nghệ Tĩnh.\n"
        "- Không diễn đạt lại quá xa câu gốc.\n"
        "- Không biến từ phổ thông thành từ khác chỉ vì nghe gần âm.\n"
        "- Ví dụ: 'hậu đậu' phải giữ là 'hậu đậu', không đổi thành 'hôi hồn', 'hú hồn' hay từ khác.\n"
        "- Ví dụ: 'mệt đứt từng khúc ruột' có thể giữ nguyên nếu đã tự nhiên.\n"
        "\n"
        "Quy tắc dùng từ điển/ngữ cảnh:\n"
        "- Ưu tiên các dòng bắt đầu bằng EXACT hơn GẦN GIỐNG.\n"
        "- Chỉ dùng GẦN GIỐNG nếu nó thật sự liên quan trực tiếp đến câu gốc.\n"
        "- Nếu GẦN GIỐNG kéo vào từ tiếng Việt phổ thông như anh, em, tình yêu, lý do, hãy bỏ qua.\n"
        "- Từ điển chỉ là gợi ý, không dịch máy móc từng từ.\n"
        "- Một từ có thể có nhiều nghĩa tùy ngữ cảnh.\n"
        "- Nếu mục có Ghi chú, hãy ưu tiên Ghi chú khi chọn nghĩa.\n"
        "- Nếu Độ tin cậy nguồn là community/needs_review, phải kiểm tra lại bằng ngữ cảnh câu.\n"
        "\n"
        "Quy tắc xử lý đa nghĩa:\n"
        "- Nếu một từ đứng cuối câu hỏi, hãy cân nhắc nó là tiểu từ hỏi như 'không?', 'à?', 'nhỉ?' thay vị đại từ xưng hô.\n"
        "- Nếu một từ đứng ở vị trí chủ ngữ hoặc tân ngữ, hãy cân nhắc nó là đại từ xưng hô như 'cậu', 'bạn', 'nó', 'hắn'.\n"
        "- Với 'ung': nếu đứng cuối câu hỏi thì thường dịch là 'không?'; nếu dùng để gọi người thì dịch là 'cậu/bạn'.\n"
        "- Với 'hấn': thường dịch là 'nó/hắn', nhưng nếu đang nói về sự vật/khái niệm như tình yêu thì có thể dịch là 'nó'.\n"
        "- Với 'rứa/ri/nớ/ni': dịch theo vị trí câu thành 'vậy/thế/này/đó/kia' cho tự nhiên.\n"
        "- Nếu 'ri/rứa' đứng sau tính từ hoặc mô tả trạng thái, thường dịch là 'thế', 'vậy', 'như thế'.\n"
        "- Ví dụ: 'hậu đậu ri' -> 'hậu đậu thế'.\n"
        "- Ví dụ: 'đẹp ri' -> 'đẹp thế'.\n"
        "- Ví dụ: 'ngu rứa' -> 'ngu thế'.\n"
        "\n"
        "Quy tắc xử lý thành ngữ, nói lái và nói trại:\n"
        "- Một số cụm phương ngữ là biến âm, nói lái hoặc nói trại của thành ngữ/cụm từ quen thuộc trong tiếng Việt phổ thông.\n"
        "- Nếu một cụm nghe gần giống một thành ngữ hoặc cách nói quen thuộc, hãy ưu tiên khôi phục về cụm phổ thông tự nhiên nhất.\n"
        "- Không dịch từng chữ nếu cụm đó thực chất là một thành ngữ hoặc cách nói cố định.\n"
        "- Ưu tiên hiểu toàn cụm trước khi hiểu từng từ riêng lẻ.\n"
        "- Nếu dịch từng chữ làm câu mất tự nhiên hoặc vô nghĩa, hãy suy luận theo cụm hoàn chỉnh.\n"
        "- Ví dụ: 'chạy thục mạ' -> 'chạy thục mạng'.\n"
        "- Ví dụ: 'bể chọ' -> 'bể bụng'.\n"
        "- Ví dụ: 'trốc tru' -> 'đầu trâu' hoặc nghĩa bóng là 'ngu'.\n"
        "- Ví dụ: 'cá tràu cá trắm' trong câu chê người có thể mang sắc thái 'trẻ trâu', 'láo láo'.\n"
        "- Ví dụ: 'troèm dứa' trong ngữ cảnh tình cảm có thể mang nghĩa 'tràn đầy'.\n"
        "- Với các cụm có tính thành ngữ, ưu tiên giữ đúng sắc thái cảm xúc thay vì bám sát từng chữ.\n"
        "\n"
        "Quy tắc giữ nguyên tiếng phổ thông:\n"
        "- Không dịch lại các từ vốn đã là tiếng phổ thông nếu chúng không phải phương ngữ trong câu.\n"
        "- Giữ nguyên tên riêng như Nam, địa danh, tên người.\n"
        "- Giữ văn phong tự nhiên, không cần bám sát từng chữ nếu làm câu bị sai nghĩa.\n"
        "- Khi không chắc một từ có phải phương ngữ không, ưu tiên giữ nguyên thay vì tự thay bằng từ khác.\n"
        "\n"
        "Quy tắc chọn mục từ dữ liệu:\n"
        "- Không bắt buộc dùng mọi mục trong từ điển/ngữ cảnh.\n"
        "- Chỉ dùng một mục nếu từ/cụm đó thật sự là phương ngữ trong câu gốc.\n"
        "- Nếu một mục EXACT là từ đơn nhưng nằm trong một cụm tiếng Việt phổ thông, hãy bỏ qua mục đó.\n"
        "- Ví dụ: 'cụ' trong 'cụ thể' không được dịch là 'cậu'.\n"
        "- Ví dụ: 'lại' trong 'em lại thích nó' không được dịch là 'lưỡi'.\n"
        "- Ưu tiên dịch theo cụm dài nhất trước, rồi mới xét từng từ đơn.\n"
        "- Nếu có nhiều nghĩa, chọn nghĩa làm cho cả câu tự nhiên và đúng ngữ pháp nhất.\n"
        "- Nếu nghĩa trong từ điển làm câu vô lý, hãy bỏ nghĩa đó và suy theo ngữ cảnh.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Từ điển/ngữ cảnh tham khảo:\n{context}\n\n"
                f"Câu gốc phương ngữ:\n{dialect_text}\n\n"
                "Hãy chuyển sang tiếng Việt phổ thông tự nhiên. Chỉ trả về một bản dịch:"
            )
        }
    ]

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages,
            temperature=0,
            top_p=1,
            max_tokens=512,
            extra_body={
                "reasoning": {
                    "enabled": False
                }
            }
        )
        translated_text = response.choices[0].message.content or ""
        return clean_translation(translated_text), matched_terms
    except Exception as e:
        print(f"[LLM] Translate error: {e}")
        return f"Lỗi dịch thuật: {str(e)}", matched_terms
