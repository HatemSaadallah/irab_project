#tr -s '[*!"#\$%&\(\)\+,\\\.\/:;<=>\?@\[\\\\]^_`\{|\}~][:space:]][:punct:][:digit:]' '\n' < $1 | sort | uniq -c | sort -nr >$1.unq
tr -s '[[:space:][:punct:][:alnum:]]' '\n' < $1 |sed 's/[،؟.,÷×:ـ]//g'| sort | uniq -c | sort -nr >$1.unq
