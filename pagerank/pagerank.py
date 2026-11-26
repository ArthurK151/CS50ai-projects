import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    num_pages = len(corpus)
    links = corpus[page]

    # If the page has no outgoing links, treat it as linking to all pages equally
    if len(links) == 0:
        for p in corpus:
            distribution[p] = 1 / num_pages
        return distribution

    # Base probability from choosing any page at random (1 - damping_factor)
    random_prob = (1 - damping_factor) / num_pages
    # Additional probability from following a link on the current page
    link_prob = damping_factor / len(links)

    for p in corpus:
        distribution[p] = random_prob
        if p in links:
            distribution[p] += link_prob

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    counts = {page: 0 for page in corpus}

    # First sample: choose a page at random with equal probability
    current_page = random.choice(list(corpus.keys()))
    counts[current_page] += 1

    # Remaining n - 1 samples: choose based on the previous page's transition model
    for _ in range(n - 1):
        model = transition_model(corpus, current_page, damping_factor)
        pages = list(model.keys())
        weights = list(model.values())
        current_page = random.choices(pages, weights=weights, k=1)[0]
        counts[current_page] += 1

    # Convert sample counts into proportions
    pageranks = {page: count / n for page, count in counts.items()}
    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    pageranks = {page: 1 / num_pages for page in corpus}

    while True:
        new_pageranks = {}

        for page in corpus:
            # First term: (1 - d) / N
            rank = (1 - damping_factor) / num_pages

            # Second term: d * sum(PR(i) / NumLinks(i)) for all pages i linking to page
            link_sum = 0
            for possible_linker, links in corpus.items():
                # Pages with no links are treated as linking to every page in the corpus
                if len(links) == 0:
                    link_sum += pageranks[possible_linker] / num_pages
                elif page in links:
                    link_sum += pageranks[possible_linker] / len(links)

            rank += damping_factor * link_sum
            new_pageranks[page] = rank

        # Check convergence: stop when no value changes by more than 0.001
        if all(abs(new_pageranks[p] - pageranks[p]) <= 0.001 for p in corpus):
            return new_pageranks

        pageranks = new_pageranks


if __name__ == "__main__":
    main()
