default:
    just --list

build:
    hatch clean && hatch build

update_version_patch:
    hatch version patch

# publish: build
#     hatch publish \
#        --user __token__ -a $PYPI_UPLOAD_TOKEN
