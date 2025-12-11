from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Tuple

import requests
import logging
import json

def articles_to_markdown(items: list[dict[str, str]]):
    markdown_lines = []
    for item in items:
        md = f"- **{item['title']}**  \n" \
            f"  - URL: {item['url']}  \n" \
            f"  - Seen: {item['seendate']}  \n" \
            f"  - Domain: {item['domain']}  \n" \
            f"  - Language: {item['language']}  \n" \
            f"  - Country: {item['sourcecountry']}"
        markdown_lines.append(md)

    return "\n".join(markdown_lines)    


class GDELTDocAPI:
    """
    Lightweight client for the GDELT DOC 2.0 API.

    This wraps the documented parameters in docs/gdelt_api_doc.txt and keeps the
    interface close to the URL schema. It uses a shared requests.Session, a tiny
    in-memory cache, and a simple rate limiter to avoid hammering the API.
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(
        self,
        *,
        rate_limit_per_minute: float | None = 60,
        cache_enabled: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        """
        Args:
            rate_limit_per_minute: Number of requests allowed per minute. Set to
                None to disable waiting between calls.
            cache_enabled: If True, reuse responses for identical parameter sets.
            session: Optional preconfigured requests.Session.
        """
        self.rate_limit_per_minute = rate_limit_per_minute
        self.cache_enabled = cache_enabled
        self.session = session or requests.Session()
        self._last_request_at: float | None = None
        self._cache: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}

    def news_search(
            self,
            queries: list[str],
            *,
            #mode: str | None = None,
            timespan: str | None = None,
            startdatetime: str | None = None,
            enddatetime: str | None = None,
            maxrecords: int | None = None,
            #sort: str | None = None,
    ) -> list[dict]:
        """
        Execute a news_search request within a certain timeframe. Search for certain entities or topics. Queries should be in English.

        Args:
            queries: The search terms, which will be joined by "OR".
            timespan: Relative lookback window like `15min`, `4h`, `1week`,
                `3m` (months), or `7days`. Use STARTDATETIME/ENDDATETIME instead
                for absolute windows.
            startdatetime: Lower bound for the search in `YYYYMMDDHHMMSS` within
                the last three months (pairs with ENDDATETIME if provided).
            enddatetime: Upper bound for the search in `YYYYMMDDHHMMSS` within the
                last three months (pairs with STARTDATETIME if provided).
            maxrecords: Number of rows for list/collage modes; defaults to 75 and
                caps at 250 (e.g. `maxrecords=150` in image collage examples).

        Examples:
            Article list of "islamic state" mentions from last week:
                news_search(queries=["islamic state"], maxrecords=100, timespan="1week")
            Article list covering wildlife crime/poaching/illegal fishing last week:
                news_search(queries=["wildlife crime"], maxrecords=100, timespan="1week")

        Returns:
            A `list[dict]` object containing articles. `raise_for_status` is invoked so HTTP
            errors propagate.
        """

        if len(queries) > 1:
            queries_joined = "(" + " OR ".join(json.dumps(q, ensure_ascii=False) for q in queries) + ")"
        else:
            queries_joined = json.dumps(queries[0], ensure_ascii=False)

        params: Dict[str, Any] = {
            "query": "sourcelang:german AND dreame" , #queries_joined + " " + "sourcelang:german",
            "mode": "artlist",
            "format": "json",
            "timespan": timespan,
            "startdatetime": startdatetime,
            "enddatetime": enddatetime,
            "maxrecords": maxrecords,
            "sort": "hybridrel",
            #"language": "german"
        }
        logging.info(f"Calling GDELTDocAPI.fetch with {params=}")


        # Drop unset parameters to keep cache keys stable and URLs clean.
        prepared_params = {k: v for k, v in params.items() if v is not None}
        cache_key = self._make_cache_key(prepared_params)

        if self.cache_enabled and cache_key in self._cache:
            return self._response_from_cache(self._cache[cache_key])

        self._respect_rate_limit()
        response = self.session.get(self.BASE_URL, params=prepared_params)
        response.raise_for_status()

        if self.cache_enabled:
            self._cache[cache_key] = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.content,
                "url": response.url,
                "encoding": response.encoding,
            }
        try:
            r_json = response.json()
            logging.info(r_json.keys())
            if "articles" not in r_json:
                return "No results found"
            return articles_to_markdown(r_json["articles"])
        except:
            logging.error(response.text)
        return f"Error({response.text.strip()})"

    def clear_cache(self) -> None:
        """Empty the in-memory response cache."""
        self._cache.clear()

    def close(self) -> None:
        """Close the underlying requests session."""
        self.session.close()

    def _respect_rate_limit(self) -> None:
        if self.rate_limit_per_minute is None:
            return

        min_interval = 60 / self.rate_limit_per_minute
        now = time.monotonic()
        if self._last_request_at is not None:
            elapsed = now - self._last_request_at
            wait_for = min_interval - elapsed
            if wait_for > 0:
                time.sleep(wait_for)
        self._last_request_at = time.monotonic()

    def _make_cache_key(self, params: Mapping[str, Any]) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
        normalized = tuple(sorted((k.lower(), str(v)) for k, v in params.items()))
        return (self.BASE_URL, normalized)

    def _response_from_cache(self, cached: Dict[str, Any]) -> requests.Response:
        response = requests.Response()
        response.status_code = cached["status_code"]
        response.headers = cached["headers"]
        response._content = cached["content"]
        response.url = cached["url"]
        response.encoding = cached["encoding"]
        return response


def test():
    import json
    api = GDELTDocAPI()
    params = {
        'queries': [
            #"Scholz arms production Germany", 
            #"Diehl Troisdorf", 
            #"Habeck defence industry minister", 
            #"Habeck Waffenminister", 
            #"Rheinmetall expansion Ukraine plant" , 
            "Dreame", 
            #"German rearmament"
        ], 
        #'mode': 'artlist', 
        #'format': 'json', 
        'timespan': '1week', 
        #'startdatetime': '20240301000000', 
        #'enddatetime': '20240331235959',
        'maxrecords': 150, 
        #'sort': 'hybridrel'
    }
    print(api.news_search(**params))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test()