#import "@preview/fletcher:0.5.8": diagram, node, edge
#set page(width: auto, height: auto, margin: 10mm, fill: white)
#set text(font: "Linux Libertine", size: 10pt)

#diagram(cell-size: (40mm, 40mm), {
  node((-1, -0.5), block(stroke: 0.5pt, inset: 8pt, radius: 4pt, [
#align(center, $a$)
#line(length: 100%)
#text(size: 9pt, `shape: (2,)`)
]), name: "0")
  node((0, 0.0), block(stroke: 0.5pt, inset: 8pt, radius: 4pt, [
#align(center, $a + b$)
#line(length: 100%)
#text(size: 9pt, `shape: (2,)`)
]), name: "1")
  node((-1, 0.5), block(stroke: 0.5pt, inset: 8pt, radius: 4pt, [
#align(center, $b$)
#line(length: 100%)
#text(size: 9pt, `shape: (2,)`)
]), name: "2")
  edge(label("0"), label("1"), "->", label: $$)
  edge(label("2"), label("1"), "->", label: $$)
})