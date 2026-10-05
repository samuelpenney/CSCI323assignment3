NFR format:
NFR# (NFR type): NFR description
NFR1 (Operational): The product shall interface with the Choice Parts System.  This provides the feed of recycled parts data.

FR format:
FR#: FR description
FR1: The user shall search for the preferred repair facility using vehicle location and radius in miles.

Dividing:
There is one blank line between two requirements.

Trace format:
Each line shows the trace links of one FR to each of the provided NFRs and there are no blank lines:
FR#,linking to NFR1,linking to NFR2,linking to NFR3
1: FR and NFR are related
0: FR and NFR are not related
FR1,0,1,0

The output of any automation shall keep the same format (i.e., the trace format).