import re
import unicodedata
from functools import lru_cache
from Levenshtein import distance as levenshtein_distance

# Configuracao global de cache LRU (ajuste aqui para benchmark)
# Use 0 para desativar o cache de uma funcao especifica.
CACHE_SIZE_DEFAULT = 100000
CACHE_SIZE_PHONETIC_KEY = 50000


# =================================================================
# 1. UTILITÁRIOS E NORMALIZAÇÃO
# =================================================================

# Mapeamento Literal (Dígito por dígito) - idêntico ao JS
LITERAL_MAP = {
    '0': ' zero ', '1': ' um ', '2': ' dois ', '3': ' três ', '4': ' quatro ',
    '5': ' cinco ', '6': ' seis ', '7': ' sete ', '8': ' oito ', '9': ' nove '
}

# Regex precompiladas
RE_DIGITS_BLOCK = re.compile(r'(\d+)')
RE_REPEAT_CHARS = re.compile(r'(.)\1+')
RE_ACCENTS = re.compile(r'[\u0300-\u036f]')
RE_NOT_ALNUM = re.compile(r'[^a-z0-9]')
RE_SPACES = re.compile(r'\s+')
RE_NOT_LETTERS = re.compile(r'[^a-z]')
RE_VOWELS = re.compile(r'[aeiou]')
RE_CE_CI = re.compile(r'ce|ci')
RE_CH = re.compile(r'ch')
RE_QU_K = re.compile(r'qu|k')
RE_XC = re.compile(r'xc')
RE_PH = re.compile(r'ph')
RE_Y = re.compile(r'y')
RE_H = re.compile(r'h')
RE_W = re.compile(r'w')


@lru_cache(maxsize=CACHE_SIZE_DEFAULT)
def number_to_words_br(n):
    """
    Conversão CARDINAL (Valor por extenso) em Português.
    Simplificação idêntica ao JS - mesmos casos especiais e fallback.
    """
    if not n or not n.isdigit():
        return n

    num = int(n)

    if num == 100:
        return " cem "
    if num == 1000000:
        return " um milhao "
    if num == 2024:
        return " dois mil e vinte quatro "

    if num < 10:
        return LITERAL_MAP[str(num)]

    if num < 20:
        dezenas = {
            10: " dez ", 11: " onze ", 12: " doze ", 13: " treze ",
            14: " quatorze ", 15: " quinze ", 16: " dezesseis ",
            17: " dezessete ", 18: " dezoito ", 19: " dezenove "
        }
        return dezenas[num]

    if num % 100 == 0 and num <= 900:
        digit = str(num // 100)
        return " " + LITERAL_MAP[digit].strip() + "centos "

    return ''.join(LITERAL_MAP.get(d, '') for d in str(n))


@lru_cache(maxsize=CACHE_SIZE_DEFAULT)
def convert_to_literal_words(s):
    """
    Via Literal: Converte dígitos para seu nome LITERAL (0 -> zero, 1 -> um)
    """
    result = s
    for digit, word in LITERAL_MAP.items():
        result = result.replace(digit, word)
    return result


@lru_cache(maxsize=CACHE_SIZE_DEFAULT)
def convert_to_cardinal_words(s):
    """
    Via Cardinal: Identifica blocos de números e converte para valor (100 -> cem)
    """
    return RE_DIGITS_BLOCK.sub(lambda m: number_to_words_br(m.group(1)), s)


@lru_cache(maxsize=CACHE_SIZE_DEFAULT)
def remove_repeated_chars(s):
    """
    Remove caracteres repetidos (ex: barbatto -> barbato)
    """
    return RE_REPEAT_CHARS.sub(r'\1', s)


@lru_cache(maxsize=CACHE_SIZE_DEFAULT)
def apply_base_normalization(s):
    """
    Normalização base (caixa baixa, acentos, caracteres especiais para espaço)
    """
    if not s:
        return ''

    s = s.lower()

    s = unicodedata.normalize('NFD', s)
    s = RE_ACCENTS.sub('', s)

    s = RE_NOT_ALNUM.sub(' ', s)
    s = RE_SPACES.sub(' ', s).strip()

    return s


@lru_cache(maxsize=CACHE_SIZE_DEFAULT)
def clean_strict(s):
    """
    Pipeline Estrito (para Levenshtein, Anagrama, Fonética)
    Passo 1: Remove repetição de caracteres ANTES de normalizar
    Passo 2: Aplica normalização base e remove espaços
    """
    clean_str = remove_repeated_chars(s)
    clean_str = apply_base_normalization(clean_str)
    return clean_str.replace(' ', '')


# =================================================================
# 2. MOTORES DE CÁLCULO (O core do algoritmo híbrido)
# =================================================================

def get_levenshtein_similarity(s1, s2):
    """
    Levenshtein (Ortografia Visual)
    Retorna: 1.0 - (distância / max(len1, len2))
    """
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    if len(s1) == 0 or len(s2) == 0:
        return 0.0

    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))

    return 1.0 - (distance / max_len)


@lru_cache(maxsize=CACHE_SIZE_PHONETIC_KEY)
def get_phonetic_key(s):
    """
    Fonética PT-BR Simplificada (Aplica removeRepeatedChars)
    """
    s = remove_repeated_chars(s)
    s = RE_NOT_LETTERS.sub('', s)

    s = RE_PH.sub('f', s)
    s = RE_Y.sub('i', s)
    s = RE_CE_CI.sub('s', s)
    s = RE_CH.sub('x', s)
    s = RE_QU_K.sub('k', s)
    s = RE_XC.sub('s', s)
    s = RE_H.sub('', s)
    s = RE_W.sub('u', s)

    if len(s) > 1:
        first = s[0]
        rest = RE_VOWELS.sub('', s[1:])
        s = first + rest

    return s.upper()


def get_phonetic_similarity(s1, s2):
    """
    Similaridade fonética entre duas strings normalizadas
    """
    code1 = ''.join(get_phonetic_key(word) for word in s1.split())
    code2 = ''.join(get_phonetic_key(word) for word in s2.split())

    return get_levenshtein_similarity(code1, code2)


def get_token_metrics(s1, s2):
    """
    Token Metrics (Jaccard, Overlap e Fuzzy)
    """
    arr1 = [x for x in s1.split() if x]
    arr2 = [x for x in s2.split() if x]

    set1 = set(arr1)
    set2 = set(arr2)

    if len(set1) == 0 and len(set2) == 0:
        return {'jaccard': 1.0, 'overlap': 1.0, 'fuzzy': 1.0}

    intersection = set1 & set2
    union = set1 | set2
    min_size = min(len(set1), len(set2))

    strict_overlap = len(intersection) / min_size if min_size > 0 else 0
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0

    fuzzy_matches = 0.0
    smaller, larger = (arr1, arr2) if len(arr1) <= len(arr2) else (arr2, arr1)

    if len(smaller) > 0:
        for t1 in smaller:
            best_match = 0.0

            if t1 in larger:
                best_match = 1.0
            else:
                ph1 = get_phonetic_key(t1)

                for t2 in larger:
                    ph2 = get_phonetic_key(t2)

                    if ph1 == ph2 and len(ph1) > 0:
                        score_ph = 1.0
                    else:
                        score_ph = get_levenshtein_similarity(ph1, ph2)

                    score_ort = get_levenshtein_similarity(t1, t2)

                    local_score = max(score_ph, score_ort)
                    if local_score > best_match:
                        best_match = local_score

            if best_match >= 0.85:
                fuzzy_matches += 1.0
            else:
                fuzzy_matches += best_match

    fuzzy_overlap = fuzzy_matches / len(smaller) if len(smaller) > 0 else 0

    return {
        'jaccard': jaccard,
        'overlap': strict_overlap,
        'fuzzy': fuzzy_overlap
    }


def get_sorted_char_similarity(m1, m2):
    """
    Anagramas (Usa cleanStrict)
    """
    sort1 = ''.join(sorted(clean_strict(m1)))
    sort2 = ''.join(sorted(clean_strict(m2)))
    return get_levenshtein_similarity(sort1, sort2)


def calculate_single_score(m1_norm, m2_norm):
    """
    Função de Cálculo do Score
    Retorna dict com final, driver, via, e vetores
    """
    clean1 = clean_strict(m1_norm)
    clean2 = clean_strict(m2_norm)

    score_orto = get_levenshtein_similarity(clean1, clean2)
    score_fon = get_phonetic_similarity(m1_norm, m2_norm)
    token_metrics = get_token_metrics(m1_norm, m2_norm)

    score_token_best = max(token_metrics['jaccard'], token_metrics['overlap'], token_metrics['fuzzy'])

    score_anagram = get_sorted_char_similarity(m1_norm, m2_norm)

    # Ajuste: o vetor de anagrama não deve ser o driver principal.
    # Ele entra apenas como sinal auxiliar, para não inflar casos
    # em que as letras coincidem, mas o sentido/sons/palavras são distintos.
    max_driver = max(score_orto, score_fon, score_token_best)
    # Exceção: quando anagrama é quase 1 (mesmas letras, ex: bionova/novabio),
    # usar como driver para não subestimar pares que só trocam ordem de sílabas.
    if score_anagram >= 0.95:
        max_driver = max(max_driver, score_anagram)
    average = (score_orto + score_fon + score_token_best + score_anagram) / 4
    final_score = (average * 0.4) + (max_driver * 0.6)

    # Sinais de similaridade forte: não aplicar penalidades que matariam casos bons.
    words1 = [w for w in m1_norm.split() if w]
    words2 = [w for w in m2_norm.split() if w]
    min_words = min(len(words1), len(words2)) if words1 and words2 else 0
    strong_orto_fon = score_orto >= 0.8 or score_fon >= 0.8
    strong_anagram = score_anagram >= 0.95
    # Contenção legítima: o lado com 1 token é o "contido" (ex: axys em axis consultoria).
    # Exige token de 4 chars, chave fonética igual e distância de Levenshtein <= 1 com algum token
    # do outro lado (axys/axis), evitando mece/miss que têm mesma chave fonética mas são distintos.
    if token_metrics['fuzzy'] >= 1.0 and min_words == 1 and (score_orto >= 0.2 or score_fon >= 0.25):
        one_token_side = words1 if len(words1) == 1 else words2
        other_side = words2 if len(words1) == 1 else words1
        single_token = one_token_side[0]
        max_other_len = max(len(t) for t in other_side) if other_side else 0
        n = len(single_token)
        ph_single = get_phonetic_key(single_token)
        ph_match = any(ph_single == get_phonetic_key(t) for t in other_side)
        # Ortografia muito próxima (1 edição) de algum token do outro lado
        lev_ok = any(levenshtein_distance(single_token, t) <= 1 for t in other_side)
        containment_ok = (
            n <= max_other_len
            and n == 4
            and ph_match
            and lev_ok
        )
    else:
        containment_ok = False

    # Penalização explícita para casos claramente distintos:
    # - nenhum token em comum (jaccard/overlap = 0)
    # - ortografia e fonética global fracas
    # Não aplicar se há similaridade forte (orto/fon altos) ou anagrama quase igual
    # ou contenção legítima (ex: axys em axis consultoria).
    if not strong_orto_fon and not strong_anagram and not containment_ok:
        if (
            token_metrics['jaccard'] == 0
            and token_metrics['overlap'] == 0
            and score_orto <= 0.5
            and score_fon <= 0.5
        ):
            final_score *= 0.1

    # Penalização adicional para casos em que:
    # - não há nenhuma palavra exatamente em comum (jaccard/overlap = 0)
    # - o vetor de tokens é fortemente impulsionado apenas por fuzzy (aproximações)
    # - a ortografia/fonética global não é muito alta (ex: MISS NINA vs MECÊ com fon=0.5)
    if not strong_orto_fon and not strong_anagram and not containment_ok:
        if (
            token_metrics['jaccard'] == 0
            and token_metrics['overlap'] == 0
            and token_metrics['fuzzy'] >= 0.9
            and max(score_orto, score_fon) < 0.55
        ):
            final_score *= 0.1

    # Ortografia global muito baixa com fuzzy alto: típico falso positivo (ex: MISS NINA vs MECÊ).
    if not containment_ok and (
        token_metrics['jaccard'] == 0
        and token_metrics['overlap'] == 0
        and token_metrics['fuzzy'] >= 0.9
        and score_orto < 0.35
    ):
        final_score *= 0.1

    # Um lado com 1 token, outro com 2+, sem palavra em comum e sem contenção legítima
    # (ex: PAZ & AMOR vs PEZÃO — fuzzy alto por coincidência, mas marcas distintas).
    if not containment_ok and (
        token_metrics['jaccard'] == 0
        and token_metrics['overlap'] == 0
        and token_metrics['fuzzy'] >= 0.9
        and min_words == 1
        and max(len(words1), len(words2)) >= 2
        and max(score_orto, score_fon) < 0.7
    ):
        final_score *= 0.1

    # Penalização complementar para casos em que:
    # - a similaridade de tokens é moderada/alta apenas via fuzzy (>= 0.5),
    # - o Jaccard é baixo (poucas palavras realmente iguais),
    # - ortografia global não passa de 50% e fonética não é extremamente alta.
    # Não aplicar quando anagrama é quase 1 (ex: bionova/novabio) ou contenção legítima (ex: axys em axis).
    if not strong_anagram and not containment_ok:
        if (
            token_metrics['jaccard'] <= 0.2
            and token_metrics['fuzzy'] >= 0.5
            and score_orto < 0.5
            and score_fon < 0.8
        ):
            final_score *= 0.1

    # Penalização forte para pares de uma única palavra em cada lado, sem nenhuma
    # palavra em comum (jaccard/overlap = 0). Não aplicar quando orto ou fon
    # são altos (ex: barbatto/barbato, caravvvela/caravela, aki/aqui, matriz/matrix,
    # pharmacia/farmácia), pois são claramente o mesmo nome ou variação.
    if not strong_orto_fon and not strong_anagram:
        if (
            token_metrics['jaccard'] == 0
            and token_metrics['overlap'] == 0
            and len(words1) == 1
            and len(words2) == 1
        ):
            final_score *= 0.1

    driver_name = 'Geral'
    if max_driver == score_fon:
        driver_name = 'Fonética (Som)'
    elif max_driver == score_orto:
        driver_name = 'Ortografia (Escrita)'
    elif max_driver == token_metrics['fuzzy'] and token_metrics['fuzzy'] > token_metrics['overlap']:
        driver_name = 'Aproximação de Termos (Fuzzy)'
    elif max_driver == token_metrics['overlap'] and token_metrics['overlap'] > token_metrics['jaccard']:
        driver_name = 'Inclusão Total de Termo'
    elif max_driver == score_token_best:
        driver_name = 'Palavras em Comum'

    return {
        'final': final_score,
        'driver': driver_name,
        'vetores': {
            'orto': score_orto,
            'fon': score_fon,
            'token': score_token_best,
            'anagram': score_anagram,
            'fuzzy': token_metrics['fuzzy']
        }
    }


# =================================================================
# 3. LÓGICA MESTRE (Com Dupla Comparação de Numerais + SIMETRIA)
# =================================================================

def calcular_score_complexo(m1, m2):
    """
    Função principal - versão simétrica
    """
    m1_norm_base = apply_base_normalization(m1)
    m2_norm_base = apply_base_normalization(m2)

    # Se não há dígitos em nenhuma das duas strings, literal/cardinal não alteram.
    # Mesmo assim, o score pode variar por direção (m1->m2 vs m2->m1),
    # então precisamos avaliar os dois sentidos e escolher o melhor,
    # reproduzindo o comportamento da lógica original das 4 vias.
    if not RE_DIGITS_BLOCK.search(m1) and not RE_DIGITS_BLOCK.search(m2):
        res_1 = calculate_single_score(m1_norm_base, m2_norm_base)
        res_1['via'] = 'Literal (dígito por dígito)'
        res_2 = calculate_single_score(m2_norm_base, m1_norm_base)
        res_2['via'] = 'Literal (dígito por dígito)'
        return res_1 if res_1['final'] >= res_2['final'] else res_2

    m1_literal = convert_to_literal_words(m1)
    m1_literal_norm = apply_base_normalization(m1_literal)
    res_1_literal = calculate_single_score(m1_literal_norm, m2_norm_base)
    res_1_literal['via'] = 'Literal (dígito por dígito)'

    m1_cardinal = convert_to_cardinal_words(m1)
    m1_cardinal_norm = apply_base_normalization(m1_cardinal)
    res_1_cardinal = calculate_single_score(m1_cardinal_norm, m2_norm_base)
    res_1_cardinal['via'] = 'Cardinal (valor por extenso)'

    m2_literal = convert_to_literal_words(m2)
    m2_literal_norm = apply_base_normalization(m2_literal)
    res_2_literal = calculate_single_score(m2_literal_norm, m1_norm_base)
    res_2_literal['via'] = 'Literal (dígito por dígito)'

    m2_cardinal = convert_to_cardinal_words(m2)
    m2_cardinal_norm = apply_base_normalization(m2_cardinal)
    res_2_cardinal = calculate_single_score(m2_cardinal_norm, m1_norm_base)
    res_2_cardinal['via'] = 'Cardinal (valor por extenso)'

    candidatos = [res_1_literal, res_1_cardinal, res_2_literal, res_2_cardinal]
    melhor = candidatos[0]

    for i in range(1, len(candidatos)):
        if candidatos[i]['final'] > melhor['final']:
            melhor = candidatos[i]

    return melhor


# =================================================================
# 4. INTERFACE PARA O PROJETO
# =================================================================

def calcular_similaridade_ofta(nome1: str, nome2: str) -> float:
    """
    Interface compatível com o projeto atual.
    Retorna score em escala 0-100 (float).
    """
    if not nome1 or not nome2:
        return 0.0
    res = calcular_score_complexo(nome1, nome2)
    return res['final'] * 100
