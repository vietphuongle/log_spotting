# Log spotting

To find and retrieve log images that are visually similar to a given query log image.

Query: a log image taken after a few weeks

Dataset: 44 registered log images

## Compilation

Edit Makefile to modify the opencv path (if necessary)

$ make

## Running

$ ./log_spotting <path/to/a/query/image>

## Example

$ ./log_spotting images/10_R2.jpg

## To register a log image

1. Crop a log image and remove ruler, label...
2. Copy the image to images folder
3. Add the image name to list_logs.txt
4. Modify the number of registered log images
