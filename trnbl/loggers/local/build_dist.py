from pathlib import Path
import base64
import requests
import json

from bs4 import BeautifulSoup, Tag


def get_remote(
	path_or_url: str,
	download_remote: bool = False,
	get_bytes: bool = False,
	allow_remote_fail: bool = True,
) -> str | bytes | None:
	"""gets a resource from a path or url

	- returns a string by default, or bytes if `get_bytes` is `True`
	- returns `None` if its from the web and `download_remote` is `False`

	# Parameters:
	 - `path_or_url : str`
	   location of the resource. if it starts with `http`, it is considered a url
	 - `download_remote : bool`
	   whether to download the resource if it is a url
	   (defaults to `False`)
	 - `get_bytes : bool`
	   whether to return the resource as bytes
	   (defaults to `False`)
	 - `allow_remote_fail : bool`
	   if a remote resource fails to download, return `None`. if this is `False`, raise an exception
	   (defaults to `True`)

	# Returns:
	 - `str|bytes|None`
	"""
	if path_or_url.startswith("http"):
		if download_remote:
			try:
				response: requests.Response = requests.get(path_or_url)
				response.raise_for_status()
			except Exception as e:
				if allow_remote_fail:
					return None
				else:
					raise e
			if get_bytes:
				return response.content
			else:
				return response.text
		else:
			return None
	else:
		path: Path = Path(path_or_url)
		if get_bytes:
			return path.read_bytes()
		else:
			return path.read_text(encoding="utf-8")


def build_dist(
	path: Path,
	minify: bool = True,
	download_remote: bool = True,
	as_json: bool = False,
) -> str:
	"""Build a single file html from a folder

	partially from https://stackoverflow.com/questions/44646481/merging-js-css-html-into-single-html
	"""
	original_html_text: str = Path(path).read_text(encoding="utf-8")
	soup: BeautifulSoup = BeautifulSoup(original_html_text, features="html.parser")

	# Find link tags. example: <link rel="stylesheet" href="css/somestyle.css">
	# also handles favicon
	for tag in soup.find_all("link", href=True):
		if tag.has_attr("href"):
			file_content: str | bytes | None = get_remote(
				tag["href"],
				download_remote=download_remote,
				get_bytes=tag.get("rel") == ["icon"], # assume text if not icon
			)

			if file_content is not None:
				# remove the tag from soup
				tag.extract()

				if tag.get("rel") == ["stylesheet"]:
					# insert style element for CSS
					new_style: Tag = soup.new_tag("style")
					new_style.string = file_content
					soup.html.head.append(new_style)
				elif tag.get("rel") == ["icon"]:
					# handle favicon
					mime_type = "image/x-icon"  # default mime type for favicon
					if tag["href"].lower().endswith(".png"):
						mime_type = "image/png"
					elif tag["href"].lower().endswith(".ico"):
						mime_type = "image/x-icon"

					base64_content = base64.b64encode(file_content).decode("ascii")
					new_link: Tag = soup.new_tag("link", rel="icon", href=f"data:{mime_type};base64,{base64_content}")
					soup.html.head.append(new_link)

	# Find script tags. example: <script src="js/somescript.js"></script>
	for tag in soup.find_all("script", src=True):
		if tag.has_attr("src"):
			file_text: str | None = get_remote(
				tag["src"],
				download_remote=download_remote,
			)

			if file_text is not None:

				# remove the tag from soup
				tag.extract()

				# insert script element
				new_script: Tag = soup.new_tag("script")
				new_script.string = file_text
				soup.html.head.append(new_script)

	# Find image tags. example: <img src="images/img1.png">
	for tag in soup.find_all("img", src=True):
		if tag.has_attr("src"):
			file_content: bytes | None = get_remote(
				tag["src"], download_remote=download_remote, get_bytes=True
			)

			if file_content is not None:
				# replace filename with base64 of the content of the file
				base64_file_content: bytes = base64.b64encode(file_content)
				tag["src"] = "data:image/png;base64, {}".format(
					base64_file_content.decode("ascii")
				)

	out_html: str = str(soup)

	if minify:
		import minify_html

		out_html = minify_html.minify(out_html, minify_css=True, minify_js=True)

	if as_json:
		out_html = json.dumps(out_html)

	return out_html


def main():
	# parse args
	import argparse

	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Build a single file HTML from a folder"
	)
	parser.add_argument("path", type=str, help="Path to the HTML file or folder")
	parser.add_argument(
		"--output", "-o", type=str, help="Output file path (default: print to console)"
	)
	parser.add_argument("--no-minify", action="store_true", help="Disable minification")
	parser.add_argument(
		"--download",
		"-d",
		action="store_true",
		help="Disable downloading remote resources",
	)
	parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

	args: argparse.Namespace = parser.parse_args()

	input_path: Path = Path(args.path)
	if not input_path.exists():
		raise FileNotFoundError(f"Path {input_path} does not exist")

	output_path = args.output or None

	# build page
	result: str = build_dist(
		path=input_path,
		minify=not args.no_minify,
		download_remote=args.download,
		as_json=args.json,
	)

	# print or save
	if output_path is None:
		print(result)
	else:
		with open(output_path, "w", encoding="utf-8") as f:
			f.write(result)


if __name__ == "__main__":
	main()
